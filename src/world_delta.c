// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

// Incremental world snapshots for rollback.
//
// The tracker keeps a mirror: one normalized byte image per world element (body, shape, contact,
// joint, island, solver set), per fixed size block of the big flat arrays (tree nodes, tree parents,
// tree proxies, pair set items) and per singleton (world scalars, id pools, sensors, graph colors,
// table counts, name count). The mirror equals the world state at the newest captured tick.
//
// A capture compares a candidate subset of images against the mirror. Each differing image pushes
// the mirror's old bytes to the undo log of the new tick and replaces the mirror image. A restore
// applies undo logs newest first, writing old images back into the world and the mirror.
//
// Candidates are what the step or the API can have touched since the last capture:
// - the neighborhood of bodies awake now and at the previous capture: the body, its shapes, its
//   contacts with both bodies and the contact edge neighbors in the other body's list, its joints,
//   its island
// - the previous capture's neighborhood, which holds elements destroyed since then along with their
//   former neighbors
// - ids allocated or freed since the last capture, found by diffing the id pools against the mirror
// - elements marked by the public API before mutation (b3GetBodyFullId, b3GetShape,
//   b3GetJointFullId), expanded with their solver set membership
// - every block of the trees and the pair set, since those are cheap to compare and hard to track
// - the event arrays: move events (b3Body::bodyMoveIndex points into them), the last step's events
//   and the end events queued by destruction since
//
// b3WorldDelta_EnableVerify compares everything and counts what the candidates missed.
//
// Images are normalized so they compare equal when the state is equal: owned heap arrays are
// inlined after the struct, their pointers nulled. Host pointers (userData, mesh, height field and
// compound data) stay verbatim since restores are in-process. Hull pointers stay verbatim too, and
// the tracker holds a hull database reference for every image that names a hull, so a hull freed by
// the world is still alive and content-identical when a restore brings its shape back.

#include "world_delta.h"

#include "bitset.h"
#include "body.h"
#include "broad_phase.h"
#include "constraint_graph.h"
#include "contact.h"
#include "container.h"
#include "core.h"
#include "id_pool.h"
#include "island.h"
#include "joint.h"
#include "name_cache.h"
#include "physics_world.h"
#include "sensor.h"
#include "shape.h"
#include "solver_set.h"
#include "table.h"

#include "box3d/box3d.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

enum
{
	b3_dtWorld,
	b3_dtCounts,
	b3_dtNames,
	b3_dtSensors,
	b3_dtColors,
	b3_dtPairHeader,
	b3_dtPairItems,
	b3_dtEvents,
	b3_dtPool0,
	b3_dtTreeHeader0 = b3_dtPool0 + 6,
	b3_dtTreeNodes0 = b3_dtTreeHeader0 + b3_bodyTypeCount,
	b3_dtTreeParents0 = b3_dtTreeNodes0 + b3_bodyTypeCount,
	b3_dtTreeProxies0 = b3_dtTreeParents0 + b3_bodyTypeCount,
	b3_dtBody = b3_dtTreeProxies0 + b3_bodyTypeCount,
	b3_dtShape,
	b3_dtContact,
	b3_dtJoint,
	b3_dtIsland,
	b3_dtSet,
	b3_dtCount
};

// Element tables, in id pool order
enum
{
	b3_deBody,
	b3_deShape,
	b3_deContact,
	b3_deJoint,
	b3_deIsland,
	b3_deSet,
	b3_deCount
};

#define B3_DELTA_NODE_BLOCK 32
#define B3_DELTA_PARENT_BLOCK 256
#define B3_DELTA_PROXY_BLOCK 64
#define B3_DELTA_PAIR_BLOCK 64
#define B3_DELTA_POOL_BLOCK 256
#define B3_DELTA_ABSENT -1
#define B3_DELTA_MAX_MISS_LOGS 16

typedef struct b3DeltaBuffer
{
	uint8_t* data;
	int size;
	int capacity;
} b3DeltaBuffer;

typedef struct b3DeltaIds
{
	int* data;
	int count;
	int capacity;
} b3DeltaIds;

// len < 0 means the slot does not exist
typedef struct b3DeltaBlob
{
	uint8_t* data;
	int len;
	int capacity;
} b3DeltaBlob;

typedef struct b3DeltaMirror
{
	b3DeltaBlob* slots;
	int count;
	int capacity;
} b3DeltaMirror;

typedef struct b3DeltaRecord
{
	int32_t table;
	int32_t index;
	int32_t len;
	int32_t pad;
} b3DeltaRecord;

typedef struct b3DeltaCounts
{
	int32_t counts[b3_deCount];
} b3DeltaCounts;

struct b3WorldDelta
{
	b3World* world;

	b3DeltaMirror mirrors[b3_dtCount];

	// logs[tick % maxTicks] holds the records that take tick back to tick - 1
	b3DeltaBuffer* logs;
	b3DeltaBuffer pending;
	int maxTicks;
	int newestTick;
	int oldestTick;

	b3DeltaBuffer scratch;

	// Candidate collection
	uint32_t stampValue;
	uint32_t* stamps[b3_deCount];
	int stampCapacity[b3_deCount];

	// Bodies expanded in this capture, plain (0) and marked (1), valid for stampValue. Topology is fixed
	// during a capture, so a second expansion adds nothing.
	uint32_t* expandStamps[2];
	int expandCapacity[2];
	b3DeltaIds candidates[b3_deCount];
	b3DeltaIds previousNeighborhood[b3_deCount];
	b3DeltaIds nextNeighborhood[b3_deCount];
	b3DeltaIds previousAwake;

	// API marks since the last capture
	b3DeltaIds marks[b3_deCount];
	b3DeltaIds markRoots[b3_deCount];
	uint32_t markEpoch;
	// Dedupe for marks and markRoots per element, valid for the current markEpoch
	uint32_t* markStamps[2][b3_deCount];
	int markStampCapacity[2][b3_deCount];
	bool markMode;

	bool verify;
	b3WorldDeltaStats stats;

	// Debug: undo bytes per table in the last capture
	int tableBytes[b3_dtCount];
	int tableCompared[b3_dtCount];
	int lastDiffOffset;
};

// Buffers

static void b3DeltaReserve( b3DeltaBuffer* buf, int capacity )
{
	if ( capacity <= buf->capacity )
	{
		return;
	}
	int newCapacity = b3MaxInt( capacity, 2 * buf->capacity );
	newCapacity = b3MaxInt( newCapacity, 256 );
	uint8_t* data = b3Alloc( newCapacity );
	if ( buf->size > 0 )
	{
		memcpy( data, buf->data, buf->size );
	}
	b3Free( buf->data, buf->capacity );
	buf->data = data;
	buf->capacity = newCapacity;
}

static void b3DeltaFreeBuffer( b3DeltaBuffer* buf )
{
	b3Free( buf->data, buf->capacity );
	*buf = (b3DeltaBuffer){ 0 };
}

static void b3DeltaPut( b3DeltaBuffer* buf, const void* src, int n )
{
	if ( n <= 0 )
	{
		return;
	}
	b3DeltaReserve( buf, buf->size + n );
	memcpy( buf->data + buf->size, src, n );
	buf->size += n;
}

static void b3DeltaPutI32( b3DeltaBuffer* buf, int v )
{
	int32_t w = v;
	b3DeltaPut( buf, &w, 4 );
}

static void b3DeltaTake( const uint8_t** cursor, void* dst, int n )
{
	if ( n <= 0 )
	{
		return;
	}
	memcpy( dst, *cursor, n );
	*cursor += n;
}

static int b3DeltaTakeI32( const uint8_t** cursor )
{
	int32_t v;
	b3DeltaTake( cursor, &v, 4 );
	return v;
}

#define b3DeltaPutArray( buf, arr )                                                                                              \
	do                                                                                                                           \
	{                                                                                                                            \
		b3DeltaPutI32( buf, ( arr ).count );                                                                                     \
		b3DeltaPut( buf, ( arr ).data, ( arr ).count * (int)sizeof( *( arr ).data ) );                                           \
	}                                                                                                                            \
	while ( 0 )

#define b3DeltaTakeArray( cursor, arr )                                                                                          \
	do                                                                                                                           \
	{                                                                                                                            \
		int takeCount = b3DeltaTakeI32( cursor );                                                                                \
		b3Array_Resize( arr, takeCount );                                                                                        \
		b3DeltaTake( cursor, ( arr ).data, takeCount * (int)sizeof( *( arr ).data ) );                                           \
	}                                                                                                                            \
	while ( 0 )

static void b3DeltaPushId( b3DeltaIds* ids, int id )
{
	if ( ids->count == ids->capacity )
	{
		int newCapacity = ids->capacity == 0 ? 64 : 2 * ids->capacity;
		int* data = b3Alloc( newCapacity * sizeof( int ) );
		if ( ids->count > 0 )
		{
			memcpy( data, ids->data, ids->count * sizeof( int ) );
		}
		b3Free( ids->data, ids->capacity * sizeof( int ) );
		ids->data = data;
		ids->capacity = newCapacity;
	}
	ids->data[ids->count++] = id;
}

static void b3DeltaCopyIds( b3DeltaIds* dst, const b3DeltaIds* src )
{
	dst->count = 0;
	for ( int i = 0; i < src->count; ++i )
	{
		b3DeltaPushId( dst, src->data[i] );
	}
}

static void b3DeltaFreeIds( b3DeltaIds* ids )
{
	b3Free( ids->data, ids->capacity * sizeof( int ) );
	*ids = (b3DeltaIds){ 0 };
}

// Stamp arrays grow on demand and start at zero, stampValue never is zero
static uint32_t* b3DeltaGrowStamps( uint32_t* stamps, int* capacity, int needed )
{
	if ( needed <= *capacity )
	{
		return stamps;
	}
	int newCapacity = b3MaxInt( needed, 2 * *capacity );
	newCapacity = b3MaxInt( newCapacity, 256 );
	stamps = b3GrowAllocZeroed( stamps, *capacity * (int)sizeof( uint32_t ), newCapacity * (int)sizeof( uint32_t ) );
	*capacity = newCapacity;
	return stamps;
}

// Hull references held by shape images

static const b3HullData* b3DeltaShapeImageHull( const uint8_t* bytes, int len, int index )
{
	if ( len < (int)sizeof( b3Shape ) )
	{
		return NULL;
	}
	b3Shape shape;
	memcpy( &shape, bytes, sizeof( b3Shape ) );
	if ( shape.id != index || shape.type != b3_hullShape )
	{
		return NULL;
	}
	return shape.hull;
}

static void b3DeltaRetain( b3WorldDelta* d, int table, int index, const uint8_t* bytes, int len )
{
	if ( table != b3_dtShape || d->world == NULL )
	{
		return;
	}
	const b3HullData* hull = b3DeltaShapeImageHull( bytes, len, index );
	if ( hull != NULL )
	{
		const b3HullData* same = b3AddHullToDatabase( d->world, hull );
		B3_ASSERT( same == hull );
		B3_UNUSED( same );
	}
}

static void b3DeltaRelease( b3WorldDelta* d, int table, int index, const uint8_t* bytes, int len )
{
	if ( table != b3_dtShape || d->world == NULL )
	{
		return;
	}
	const b3HullData* hull = b3DeltaShapeImageHull( bytes, len, index );
	if ( hull != NULL )
	{
		b3RemoveHullFromDatabase( d->world, hull );
	}
}

// Mirror

static b3DeltaBlob* b3DeltaMirrorSlot( b3WorldDelta* d, int table, int index )
{
	b3DeltaMirror* m = d->mirrors + table;
	if ( index >= m->capacity )
	{
		int newCapacity = b3MaxInt( index + 1, 2 * m->capacity );
		newCapacity = b3MaxInt( newCapacity, 16 );
		b3DeltaBlob* slots = b3Alloc( newCapacity * sizeof( b3DeltaBlob ) );
		if ( m->capacity > 0 )
		{
			memcpy( slots, m->slots, m->capacity * sizeof( b3DeltaBlob ) );
		}
		for ( int i = m->capacity; i < newCapacity; ++i )
		{
			slots[i] = (b3DeltaBlob){ NULL, B3_DELTA_ABSENT, 0 };
		}
		b3Free( m->slots, m->capacity * sizeof( b3DeltaBlob ) );
		m->slots = slots;
		m->capacity = newCapacity;
	}
	if ( index >= m->count )
	{
		m->count = index + 1;
	}
	return m->slots + index;
}

static void b3DeltaBlobSet( b3WorldDelta* d, b3DeltaBlob* blob, const uint8_t* bytes, int len )
{
	if ( len > blob->capacity )
	{
		d->stats.mirrorBytes += len - blob->capacity;
		b3Free( blob->data, blob->capacity );
		blob->data = b3Alloc( len );
		blob->capacity = len;
	}
	if ( len > 0 )
	{
		memcpy( blob->data, bytes, len );
	}
	blob->len = len;
}

// Logs

static void b3DeltaAppendRecord( b3DeltaBuffer* log, int table, int index, const uint8_t* bytes, int len )
{
	int payload = len > 0 ? len : 0;
	int padded = ( payload + 7 ) & ~7;
	b3DeltaReserve( log, log->size + (int)sizeof( b3DeltaRecord ) + padded );
	b3DeltaRecord record = { table, index, len, 0 };
	memcpy( log->data + log->size, &record, sizeof( record ) );
	log->size += (int)sizeof( record );
	if ( payload > 0 )
	{
		memcpy( log->data + log->size, bytes, payload );
	}
	log->size += padded;
}

static const uint8_t* b3DeltaNextRecord( const uint8_t* p, b3DeltaRecord* record )
{
	memcpy( record, p, sizeof( b3DeltaRecord ) );
	int payload = record->len > 0 ? record->len : 0;
	return p + sizeof( b3DeltaRecord ) + ( ( payload + 7 ) & ~7 );
}

static void b3DeltaDropLog( b3WorldDelta* d, b3DeltaBuffer* log )
{
	const uint8_t* p = log->data;
	const uint8_t* end = log->data + log->size;
	while ( p < end )
	{
		b3DeltaRecord record;
		const uint8_t* next = b3DeltaNextRecord( p, &record );
		b3DeltaRelease( d, record.table, record.index, p + sizeof( b3DeltaRecord ), record.len );
		p = next;
	}
	d->stats.logBytes -= log->size;
	log->size = 0;
}

// Slot counts

static b3DynamicTree* b3DeltaTree( b3World* world, int table, int base )
{
	return world->broadPhase.trees + ( table - base );
}

static int b3DeltaBlockCount( int count, int block )
{
	return ( count + block - 1 ) / block;
}

static b3IdPool* b3DeltaPool( b3World* world, int element );

static int b3DeltaSlotCount( b3World* world, int table )
{
	switch ( table )
	{
		case b3_dtColors:
			return B3_GRAPH_COLOR_COUNT;
		case b3_dtPairItems:
			return b3DeltaBlockCount( (int)world->broadPhase.pairSet.capacity, B3_DELTA_PAIR_BLOCK );
		case b3_dtBody:
			return world->bodies.count;
		case b3_dtShape:
			return world->shapes.count;
		case b3_dtContact:
			return world->contacts.count;
		case b3_dtJoint:
			return world->joints.count;
		case b3_dtIsland:
			return world->islands.count;
		case b3_dtSet:
			return world->solverSets.count;
		default:
			break;
	}

	if ( b3_dtPool0 <= table && table < b3_dtPool0 + b3_deCount )
	{
		return 1 + b3DeltaBlockCount( b3DeltaPool( world, table - b3_dtPool0 )->freeArray.count, B3_DELTA_POOL_BLOCK );
	}
	if ( b3_dtTreeNodes0 <= table && table < b3_dtTreeNodes0 + b3_bodyTypeCount )
	{
		return b3DeltaBlockCount( b3DeltaTree( world, table, b3_dtTreeNodes0 )->nodeEnd, B3_DELTA_NODE_BLOCK );
	}
	if ( b3_dtTreeParents0 <= table && table < b3_dtTreeParents0 + b3_bodyTypeCount )
	{
		return b3DeltaBlockCount( b3DeltaTree( world, table, b3_dtTreeParents0 )->nodeEnd, B3_DELTA_PARENT_BLOCK );
	}
	if ( b3_dtTreeProxies0 <= table && table < b3_dtTreeProxies0 + b3_bodyTypeCount )
	{
		return b3DeltaBlockCount( b3DeltaTree( world, table, b3_dtTreeProxies0 )->proxyCapacity, B3_DELTA_PROXY_BLOCK );
	}

	return 1;
}

static b3IdPool* b3DeltaPool( b3World* world, int element )
{
	switch ( element )
	{
		case b3_deBody:
			return &world->bodyIdPool;
		case b3_deShape:
			return &world->shapeIdPool;
		case b3_deContact:
			return &world->contactIdPool;
		case b3_deJoint:
			return &world->jointIdPool;
		case b3_deIsland:
			return &world->islandIdPool;
		default:
			return &world->solverSetIdPool;
	}
}

// Images. Returns the image bytes and length, or NULL with B3_DELTA_ABSENT for a missing slot.
// Flat POD slots point straight into the world, the rest are packed into the scratch buffer.

static const uint8_t* b3DeltaBlock( const void* base, int elementSize, int count, int block, int index, int* len )
{
	int first = index * block;
	int last = b3MinInt( first + block, count );
	*len = ( last - first ) * elementSize;
	return (const uint8_t*)base + (size_t)first * elementSize;
}

static void b3DeltaPackWorld( b3DeltaBuffer* s, b3World* world )
{
	b3DeltaPut( s, &world->gravity, sizeof( b3Vec3 ) );
	b3DeltaPut( s, &world->hitEventThreshold, sizeof( float ) );
	b3DeltaPut( s, &world->restitutionThreshold, sizeof( float ) );
	b3DeltaPut( s, &world->restitutionIterations, sizeof( int ) );
	b3DeltaPut( s, &world->maxLinearSpeed, sizeof( float ) );
	b3DeltaPut( s, &world->contactSpeed, sizeof( float ) );
	b3DeltaPut( s, &world->contactHertz, sizeof( float ) );
	b3DeltaPut( s, &world->contactDampingRatio, sizeof( float ) );
	b3DeltaPut( s, &world->contactRecycleDistance, sizeof( float ) );
	b3DeltaPut( s, &world->stepIndex, sizeof( uint64_t ) );
	b3DeltaPut( s, &world->splitIslandId, sizeof( int ) );
	b3DeltaPut( s, &world->inv_h, sizeof( float ) );
	b3DeltaPut( s, &world->inv_dt, sizeof( float ) );
	b3DeltaPut( s, &world->endEventArrayIndex, sizeof( int ) );
	b3DeltaPut( s, &world->compoundShapeCount, sizeof( int ) );
	b3DeltaPut( s, &world->maxCapacity, sizeof( b3Capacity ) );

	// Callbacks and contexts are host pointers, valid since restores are in-process
	b3DeltaPut( s, &world->frictionCallback, sizeof( world->frictionCallback ) );
	b3DeltaPut( s, &world->restitutionCallback, sizeof( world->restitutionCallback ) );
	b3DeltaPut( s, &world->preSolveFcn, sizeof( world->preSolveFcn ) );
	b3DeltaPut( s, &world->preSolveContext, sizeof( void* ) );
	b3DeltaPut( s, &world->customFilterFcn, sizeof( world->customFilterFcn ) );
	b3DeltaPut( s, &world->customFilterContext, sizeof( void* ) );
	b3DeltaPut( s, &world->userData, sizeof( void* ) );

	uint8_t flags[5] = { world->enableSleep, world->enableWarmStarting, world->enableContinuous, world->enableSpeculative,
						 world->enableRestitutionPropagation };
	b3DeltaPut( s, flags, sizeof( flags ) );
}

static void b3DeltaUnpackWorld( const uint8_t* p, b3World* world )
{
	b3DeltaTake( &p, &world->gravity, sizeof( b3Vec3 ) );
	b3DeltaTake( &p, &world->hitEventThreshold, sizeof( float ) );
	b3DeltaTake( &p, &world->restitutionThreshold, sizeof( float ) );
	b3DeltaTake( &p, &world->restitutionIterations, sizeof( int ) );
	b3DeltaTake( &p, &world->maxLinearSpeed, sizeof( float ) );
	b3DeltaTake( &p, &world->contactSpeed, sizeof( float ) );
	b3DeltaTake( &p, &world->contactHertz, sizeof( float ) );
	b3DeltaTake( &p, &world->contactDampingRatio, sizeof( float ) );
	b3DeltaTake( &p, &world->contactRecycleDistance, sizeof( float ) );
	b3DeltaTake( &p, &world->stepIndex, sizeof( uint64_t ) );
	b3DeltaTake( &p, &world->splitIslandId, sizeof( int ) );
	b3DeltaTake( &p, &world->inv_h, sizeof( float ) );
	b3DeltaTake( &p, &world->inv_dt, sizeof( float ) );
	b3DeltaTake( &p, &world->endEventArrayIndex, sizeof( int ) );
	b3DeltaTake( &p, &world->compoundShapeCount, sizeof( int ) );
	b3DeltaTake( &p, &world->maxCapacity, sizeof( b3Capacity ) );
	b3DeltaTake( &p, &world->frictionCallback, sizeof( world->frictionCallback ) );
	b3DeltaTake( &p, &world->restitutionCallback, sizeof( world->restitutionCallback ) );
	b3DeltaTake( &p, &world->preSolveFcn, sizeof( world->preSolveFcn ) );
	b3DeltaTake( &p, &world->preSolveContext, sizeof( void* ) );
	b3DeltaTake( &p, &world->customFilterFcn, sizeof( world->customFilterFcn ) );
	b3DeltaTake( &p, &world->customFilterContext, sizeof( void* ) );
	b3DeltaTake( &p, &world->userData, sizeof( void* ) );
	uint8_t flags[5];
	b3DeltaTake( &p, flags, sizeof( flags ) );
	world->enableSleep = flags[0];
	world->enableWarmStarting = flags[1];
	world->enableContinuous = flags[2];
	world->enableSpeculative = flags[3];
	world->enableRestitutionPropagation = flags[4];
}

static void b3DeltaPackBitSet( b3DeltaBuffer* s, const b3BitSet* bits )
{
	// The whole capacity travels: b3GrowBitSet exposes words above blockCount without clearing them
	b3DeltaPutI32( s, (int)bits->blockCount );
	b3DeltaPutI32( s, (int)bits->blockCapacity );
	b3DeltaPut( s, bits->bits, (int)( bits->blockCapacity * sizeof( uint64_t ) ) );
}

static void b3DeltaUnpackBitSet( const uint8_t** p, b3BitSet* bits )
{
	uint32_t blockCount = (uint32_t)b3DeltaTakeI32( p );
	uint32_t blockCapacity = (uint32_t)b3DeltaTakeI32( p );
	if ( blockCapacity != bits->blockCapacity )
	{
		b3Free( bits->bits, bits->blockCapacity * sizeof( uint64_t ) );
		bits->bits = blockCapacity > 0 ? b3Alloc( blockCapacity * sizeof( uint64_t ) ) : NULL;
		bits->blockCapacity = blockCapacity;
	}
	bits->blockCount = blockCount;
	b3DeltaTake( p, bits->bits, (int)( blockCapacity * sizeof( uint64_t ) ) );
}

static const uint8_t* b3DeltaImage( b3WorldDelta* d, int table, int index, int* len )
{
	b3World* world = d->world;
	if ( index >= b3DeltaSlotCount( world, table ) )
	{
		*len = B3_DELTA_ABSENT;
		return NULL;
	}

	b3DeltaBuffer* s = &d->scratch;
	s->size = 0;

	switch ( table )
	{
		case b3_dtWorld:
			b3DeltaPackWorld( s, world );
			break;

		case b3_dtCounts:
		{
			b3DeltaCounts counts;
			for ( int i = 0; i < b3_deCount; ++i )
			{
				counts.counts[i] = b3DeltaSlotCount( world, b3_dtBody + i );
			}
			b3DeltaPut( s, &counts, sizeof( counts ) );
			break;
		}

		case b3_dtNames:
			// Names are append only, the count is enough to truncate
			b3DeltaPutI32( s, world->names.entries.count );
			break;

		case b3_dtSensors:
			b3DeltaPutI32( s, world->sensors.count );
			for ( int i = 0; i < world->sensors.count; ++i )
			{
				b3Sensor* sensor = world->sensors.data + i;
				b3DeltaPutI32( s, sensor->shapeId );
				b3DeltaPutArray( s, sensor->hits );
				b3DeltaPutArray( s, sensor->overlaps1 );
				b3DeltaPutArray( s, sensor->overlaps2 );
			}
			break;

		case b3_dtColors:
		{
			b3GraphColor* color = world->constraintGraph.colors + index;
			b3DeltaPackBitSet( s, &color->bodySet );
			b3DeltaPutArray( s, color->jointSims );
			b3DeltaPutArray( s, color->convexContacts );
			b3DeltaPutArray( s, color->contacts );
			break;
		}

		case b3_dtPairHeader:
			b3DeltaPutI32( s, (int)world->broadPhase.pairSet.capacity );
			b3DeltaPutI32( s, (int)world->broadPhase.pairSet.count );
			break;

		case b3_dtEvents:
			// b3Body::bodyMoveIndex points into the last step's move events. The other arrays hold the
			// last step's events for the getters, and the current end buffer queues events from
			// destruction until the next step publishes them.
			b3DeltaPutArray( s, world->bodyMoveEvents );
			b3DeltaPutArray( s, world->sensorBeginEvents );
			b3DeltaPutArray( s, world->contactBeginEvents );
			for ( int i = 0; i < 2; ++i )
			{
				b3DeltaPutArray( s, world->sensorEndEvents[i] );
				b3DeltaPutArray( s, world->contactEndEvents[i] );
			}
			b3DeltaPutArray( s, world->contactHitEvents );
			b3DeltaPutArray( s, world->jointEvents );
			break;

		case b3_dtPairItems:
		{
			b3HashSet* set = &world->broadPhase.pairSet;
			return b3DeltaBlock( set->items, sizeof( b3SetItem ), (int)set->capacity, B3_DELTA_PAIR_BLOCK, index, len );
		}

		case b3_dtBody:
			*len = sizeof( b3Body );
			return (const uint8_t*)( world->bodies.data + index );

		case b3_dtShape:
		{
			b3Shape* src = world->shapes.data + index;
			b3Shape shape = *src;
			bool isLive = shape.id == index;
			shape.materials = NULL;
			shape.userShape = NULL;
			if ( isLive == false )
			{
				memset( &shape.capsule, 0, sizeof( shape.capsule ) );
			}
			b3DeltaPut( s, &shape, sizeof( b3Shape ) );
			b3DeltaPut( s, world->fatAABBs.data + index, sizeof( b3AABB ) );
			int materialCount = isLive && src->materials != NULL ? src->materialCount : 0;
			b3DeltaPutI32( s, materialCount );
			b3DeltaPut( s, src->materials, materialCount * (int)sizeof( b3SurfaceMaterial ) );
			break;
		}

		case b3_dtContact:
		{
			b3Contact* src = world->contacts.data + index;
			b3Contact contact = *src;
			bool isLive = contact.contactId == index;
			contact.manifolds = NULL;
			if ( contact.flags & b3_simMeshContact )
			{
				contact.meshContact.triangleCache.data = NULL;
				contact.meshContact.triangleCache.count = 0;
				contact.meshContact.triangleCache.capacity = 0;
			}
			b3DeltaPut( s, &contact, sizeof( b3Contact ) );
			if ( isLive )
			{
				int manifoldCount = src->manifolds != NULL ? src->manifoldCount : 0;
				b3DeltaPutI32( s, manifoldCount );
				b3DeltaPut( s, src->manifolds, manifoldCount * (int)sizeof( b3Manifold ) );
				if ( src->flags & b3_simMeshContact )
				{
					b3DeltaPutArray( s, src->meshContact.triangleCache );
				}
			}
			break;
		}

		case b3_dtJoint:
			*len = sizeof( b3Joint );
			return (const uint8_t*)( world->joints.data + index );

		case b3_dtIsland:
		{
			b3Island* island = world->islands.data + index;
			b3DeltaPutI32( s, island->setIndex );
			b3DeltaPutI32( s, island->localIndex );
			b3DeltaPutI32( s, island->islandId );
			b3DeltaPutI32( s, island->constraintRemoveCount );
			b3DeltaPutArray( s, island->bodies );
			b3DeltaPutArray( s, island->contacts );
			b3DeltaPutArray( s, island->joints );
			break;
		}

		case b3_dtSet:
		{
			b3SolverSet* set = world->solverSets.data + index;
			b3DeltaPutI32( s, set->setIndex );
			b3DeltaPutArray( s, set->bodySims );
			b3DeltaPutArray( s, set->bodyStates );
			b3DeltaPutArray( s, set->jointSims );
			b3DeltaPutArray( s, set->contactIndices );
			b3DeltaPutArray( s, set->islandSims );
			break;
		}

		default:
		{
			if ( b3_dtPool0 <= table && table < b3_dtPool0 + b3_deCount )
			{
				// Slot 0 is the header, then blocks of the free stack so a push or pop logs one block
				b3IdPool* pool = b3DeltaPool( world, table - b3_dtPool0 );
				if ( index > 0 )
				{
					return b3DeltaBlock( pool->freeArray.data, sizeof( int ), pool->freeArray.count, B3_DELTA_POOL_BLOCK, index - 1,
										 len );
				}
				b3DeltaPutI32( s, pool->nextIndex );
				b3DeltaPutI32( s, pool->freeArray.count );
				break;
			}
			if ( b3_dtTreeHeader0 <= table && table < b3_dtTreeHeader0 + b3_bodyTypeCount )
			{
				b3DynamicTree* tree = b3DeltaTree( world, table, b3_dtTreeHeader0 );
				b3DeltaPut( s, &tree->version, sizeof( uint64_t ) );
				b3DeltaPutI32( s, tree->nodeEnd );
				b3DeltaPutI32( s, tree->pairFreeList );
				b3DeltaPutI32( s, tree->proxyCount );
				b3DeltaPutI32( s, tree->proxyCapacity );
				b3DeltaPutI32( s, tree->proxyFreeList );
				b3DeltaPutI32( s, tree->dfsOrdered ? 1 : 0 );
				break;
			}
			if ( b3_dtTreeNodes0 <= table && table < b3_dtTreeNodes0 + b3_bodyTypeCount )
			{
				b3DynamicTree* tree = b3DeltaTree( world, table, b3_dtTreeNodes0 );
				return b3DeltaBlock( tree->nodes, sizeof( b3TreeNode ), tree->nodeEnd, B3_DELTA_NODE_BLOCK, index, len );
			}
			if ( b3_dtTreeParents0 <= table && table < b3_dtTreeParents0 + b3_bodyTypeCount )
			{
				b3DynamicTree* tree = b3DeltaTree( world, table, b3_dtTreeParents0 );
				return b3DeltaBlock( tree->parents, sizeof( int32_t ), tree->nodeEnd, B3_DELTA_PARENT_BLOCK, index, len );
			}
			if ( b3_dtTreeProxies0 <= table && table < b3_dtTreeProxies0 + b3_bodyTypeCount )
			{
				b3DynamicTree* tree = b3DeltaTree( world, table, b3_dtTreeProxies0 );
				return b3DeltaBlock( tree->proxies, sizeof( b3TreeProxy ), tree->proxyCapacity, B3_DELTA_PROXY_BLOCK, index,
									 len );
			}
			B3_ASSERT( false );
			break;
		}
	}

	*len = s->size;
	return s->data;
}

// Apply an image to the world. Element and block slots must exist, the restore grows the tables
// first. An absent image releases the slot's heap and zeroes it; the restore truncates afterwards.

static void* b3DeltaResizeExact( void* mem, int oldSize, int newSize )
{
	void* newMem = newSize > 0 ? b3Alloc( newSize ) : NULL;
	int keep = b3MinInt( oldSize, newSize );
	if ( keep > 0 )
	{
		memcpy( newMem, mem, keep );
	}
	if ( newSize > keep )
	{
		memset( (uint8_t*)newMem + keep, 0, newSize - keep );
	}
	b3Free( mem, oldSize );
	return newMem;
}

static void b3DeltaFreeShapeHeap( b3World* world, b3Shape* shape )
{
	if ( shape->materials != NULL )
	{
		b3Free( shape->materials, shape->materialCount * sizeof( b3SurfaceMaterial ) );
		shape->materials = NULL;
	}
	if ( shape->type == b3_hullShape && shape->hull != NULL )
	{
		b3RemoveHullFromDatabase( world, shape->hull );
		shape->hull = NULL;
	}
}

static void b3DeltaApplyShape( b3World* world, int index, const uint8_t* p, int len )
{
	b3Shape* dst = world->shapes.data + index;
	bool wasLive = dst->id == index;
	void* userShape = wasLive ? dst->userShape : NULL;
	uint16_t oldGeneration = dst->generation;

	// The geometry is the union at the tail of b3Shape
	enum
	{
		geometryOffset = offsetof( b3Shape, capsule ),
		geometrySize = sizeof( b3Shape ) - offsetof( b3Shape, capsule )
	};
	b3ShapeType oldType = dst->type;
	uint8_t oldGeometry[geometrySize];
	memcpy( oldGeometry, (const uint8_t*)dst + geometryOffset, geometrySize );

	if ( wasLive )
	{
		b3DeltaFreeShapeHeap( world, dst );
	}

	if ( len == B3_DELTA_ABSENT )
	{
		if ( userShape != NULL && world->destroyDebugShape != NULL )
		{
			world->destroyDebugShape( userShape, world->userDebugShapeContext );
		}
		memset( dst, 0, sizeof( b3Shape ) );
		return;
	}

	b3DeltaTake( &p, dst, sizeof( b3Shape ) );
	b3DeltaTake( &p, world->fatAABBs.data + index, sizeof( b3AABB ) );
	int materialCount = b3DeltaTakeI32( &p );
	bool isLive = dst->id == index;

	// Keep the renderer handle when the same shape still occupies the slot with the same geometry.
	// Geometry setters keep the generation and drop the handle, so a restore across one must too.
	bool sameGeometry = dst->type == oldType && memcmp( oldGeometry, (const uint8_t*)dst + geometryOffset, geometrySize ) == 0;
	if ( isLive && dst->generation == oldGeneration && sameGeometry )
	{
		dst->userShape = userShape;
	}
	else if ( userShape != NULL && world->destroyDebugShape != NULL )
	{
		world->destroyDebugShape( userShape, world->userDebugShapeContext );
	}

	if ( materialCount > 0 )
	{
		dst->materials = b3Alloc( materialCount * sizeof( b3SurfaceMaterial ) );
		b3DeltaTake( &p, dst->materials, materialCount * (int)sizeof( b3SurfaceMaterial ) );
	}

	if ( isLive && dst->type == b3_hullShape && dst->hull != NULL )
	{
		// The tracker holds a reference, so the hull is still in the database
		const b3HullData* same = b3AddHullToDatabase( world, dst->hull );
		B3_ASSERT( same == dst->hull );
		B3_UNUSED( same );
	}
}

static void b3DeltaApplyContact( b3World* world, int index, const uint8_t* p, int len )
{
	b3Contact* dst = world->contacts.data + index;
	if ( dst->contactId == index )
	{
		if ( dst->manifolds != NULL )
		{
			b3FreeManifolds( world, dst->manifolds, dst->manifoldCount );
			dst->manifolds = NULL;
		}
		if ( dst->flags & b3_simMeshContact )
		{
			b3Array_Destroy( dst->meshContact.triangleCache );
		}
	}

	if ( len == B3_DELTA_ABSENT )
	{
		memset( dst, 0, sizeof( b3Contact ) );
		return;
	}

	b3DeltaTake( &p, dst, sizeof( b3Contact ) );
	if ( dst->contactId != index )
	{
		return;
	}

	int manifoldCount = b3DeltaTakeI32( &p );
	if ( manifoldCount > 0 )
	{
		dst->manifolds = b3AllocateManifolds( world, manifoldCount );
		b3DeltaTake( &p, dst->manifolds, manifoldCount * (int)sizeof( b3Manifold ) );
	}
	if ( dst->flags & b3_simMeshContact )
	{
		b3Array_Create( dst->meshContact.triangleCache );
		b3DeltaTakeArray( &p, dst->meshContact.triangleCache );
	}
}

static void b3DeltaApplyIsland( b3World* world, int index, const uint8_t* p, int len )
{
	b3Island* island = world->islands.data + index;
	if ( len == B3_DELTA_ABSENT )
	{
		b3Array_Destroy( island->bodies );
		b3Array_Destroy( island->contacts );
		b3Array_Destroy( island->joints );
		memset( island, 0, sizeof( b3Island ) );
		return;
	}
	island->setIndex = b3DeltaTakeI32( &p );
	island->localIndex = b3DeltaTakeI32( &p );
	island->islandId = b3DeltaTakeI32( &p );
	island->constraintRemoveCount = b3DeltaTakeI32( &p );
	b3DeltaTakeArray( &p, island->bodies );
	b3DeltaTakeArray( &p, island->contacts );
	b3DeltaTakeArray( &p, island->joints );

	// A freed slot owns no arrays: Box3D frees them on destroy and creates fresh ones on reuse
	if ( island->setIndex == B3_NULL_INDEX )
	{
		b3Array_Destroy( island->bodies );
		b3Array_Destroy( island->contacts );
		b3Array_Destroy( island->joints );
	}
}

static void b3DeltaApplySet( b3World* world, int index, const uint8_t* p, int len )
{
	b3SolverSet* set = world->solverSets.data + index;
	if ( len == B3_DELTA_ABSENT )
	{
		b3Array_Destroy( set->bodySims );
		b3Array_Destroy( set->bodyStates );
		b3Array_Destroy( set->jointSims );
		b3Array_Destroy( set->contactIndices );
		b3Array_Destroy( set->islandSims );
		memset( set, 0, sizeof( b3SolverSet ) );
		return;
	}
	set->setIndex = b3DeltaTakeI32( &p );
	b3DeltaTakeArray( &p, set->bodySims );
	b3DeltaTakeArray( &p, set->bodyStates );
	b3DeltaTakeArray( &p, set->jointSims );
	b3DeltaTakeArray( &p, set->contactIndices );
	b3DeltaTakeArray( &p, set->islandSims );

	if ( set->setIndex == B3_NULL_INDEX )
	{
		b3Array_Destroy( set->bodySims );
		b3Array_Destroy( set->bodyStates );
		b3Array_Destroy( set->jointSims );
		b3Array_Destroy( set->contactIndices );
		b3Array_Destroy( set->islandSims );
	}
}

static void b3DeltaApplySensors( b3World* world, const uint8_t* p )
{
	int count = b3DeltaTakeI32( &p );
	for ( int i = count; i < world->sensors.count; ++i )
	{
		b3Sensor* sensor = world->sensors.data + i;
		b3Array_Destroy( sensor->hits );
		b3Array_Destroy( sensor->overlaps1 );
		b3Array_Destroy( sensor->overlaps2 );
	}
	int oldCount = world->sensors.count;
	b3Array_Resize( world->sensors, count );
	for ( int i = oldCount; i < count; ++i )
	{
		memset( world->sensors.data + i, 0, sizeof( b3Sensor ) );
	}
	for ( int i = 0; i < count; ++i )
	{
		b3Sensor* sensor = world->sensors.data + i;
		sensor->shapeId = b3DeltaTakeI32( &p );
		b3DeltaTakeArray( &p, sensor->hits );
		b3DeltaTakeArray( &p, sensor->overlaps1 );
		b3DeltaTakeArray( &p, sensor->overlaps2 );
	}
}

static void b3DeltaApplyTreeHeader( b3DynamicTree* tree, const uint8_t* p )
{
	uint64_t version;
	b3DeltaTake( &p, &version, sizeof( uint64_t ) );
	int nodeEnd = b3DeltaTakeI32( &p );
	int pairFreeList = b3DeltaTakeI32( &p );
	int proxyCount = b3DeltaTakeI32( &p );
	int proxyCapacity = b3DeltaTakeI32( &p );
	int proxyFreeList = b3DeltaTakeI32( &p );
	bool dfsOrdered = b3DeltaTakeI32( &p ) != 0;

	if ( nodeEnd > tree->nodeCapacity )
	{
		// Same as the bump allocator growth: the spare must match, the rebuild allocates it again
		tree->nodes = B3_GROW_ZERO( tree->nodes, tree->nodeCapacity, nodeEnd );
		tree->parents = B3_GROW_ZERO( tree->parents, tree->nodeCapacity, nodeEnd );
		b3Free( tree->swapNodes, tree->nodeCapacity * sizeof( b3TreeNode ) );
		tree->swapNodes = NULL;
		tree->nodeCapacity = nodeEnd;
	}

	if ( proxyCapacity != tree->proxyCapacity )
	{
		tree->proxies = b3DeltaResizeExact( tree->proxies, tree->proxyCapacity * (int)sizeof( b3TreeProxy ),
											proxyCapacity * (int)sizeof( b3TreeProxy ) );
		tree->proxyCapacity = proxyCapacity;
	}

	tree->version = version;
	tree->nodeEnd = nodeEnd;
	tree->pairFreeList = pairFreeList;
	tree->proxyCount = proxyCount;
	tree->proxyFreeList = proxyFreeList;
	tree->dfsOrdered = dfsOrdered;
}

static void b3DeltaApplyBlock( void* base, int elementSize, int block, int index, const uint8_t* p, int len )
{
	if ( len > 0 )
	{
		memcpy( (uint8_t*)base + (size_t)index * block * elementSize, p, len );
	}
}

static void b3DeltaApply( b3WorldDelta* d, int table, int index, const uint8_t* p, int len )
{
	b3World* world = d->world;

	switch ( table )
	{
		case b3_dtWorld:
			b3DeltaUnpackWorld( p, world );
			return;

		case b3_dtCounts:
			// Table sizes are set by the restore around the element records
			return;

		case b3_dtNames:
			b3TruncateNames( &world->names, b3DeltaTakeI32( &p ) );
			return;

		case b3_dtSensors:
			b3DeltaApplySensors( world, p );
			return;

		case b3_dtColors:
		{
			b3GraphColor* color = world->constraintGraph.colors + index;
			b3DeltaUnpackBitSet( &p, &color->bodySet );
			b3DeltaTakeArray( &p, color->jointSims );
			b3DeltaTakeArray( &p, color->convexContacts );
			b3DeltaTakeArray( &p, color->contacts );
			return;
		}

		case b3_dtPairHeader:
		{
			b3HashSet* set = &world->broadPhase.pairSet;
			uint32_t capacity = (uint32_t)b3DeltaTakeI32( &p );
			uint32_t count = (uint32_t)b3DeltaTakeI32( &p );
			if ( capacity != set->capacity )
			{
				set->items = b3DeltaResizeExact( set->items, (int)( set->capacity * sizeof( b3SetItem ) ),
												 (int)( capacity * sizeof( b3SetItem ) ) );
				set->capacity = capacity;
			}
			set->count = count;
			return;
		}

		case b3_dtEvents:
			b3DeltaTakeArray( &p, world->bodyMoveEvents );
			b3DeltaTakeArray( &p, world->sensorBeginEvents );
			b3DeltaTakeArray( &p, world->contactBeginEvents );
			for ( int i = 0; i < 2; ++i )
			{
				b3DeltaTakeArray( &p, world->sensorEndEvents[i] );
				b3DeltaTakeArray( &p, world->contactEndEvents[i] );
			}
			b3DeltaTakeArray( &p, world->contactHitEvents );
			b3DeltaTakeArray( &p, world->jointEvents );
			return;

		case b3_dtPairItems:
			b3DeltaApplyBlock( world->broadPhase.pairSet.items, sizeof( b3SetItem ), B3_DELTA_PAIR_BLOCK, index, p, len );
			return;

		case b3_dtBody:
			if ( len == B3_DELTA_ABSENT )
			{
				memset( world->bodies.data + index, 0, sizeof( b3Body ) );
			}
			else
			{
				memcpy( world->bodies.data + index, p, sizeof( b3Body ) );
			}
			return;

		case b3_dtShape:
			b3DeltaApplyShape( world, index, p, len );
			return;

		case b3_dtContact:
			b3DeltaApplyContact( world, index, p, len );
			return;

		case b3_dtJoint:
			if ( len == B3_DELTA_ABSENT )
			{
				memset( world->joints.data + index, 0, sizeof( b3Joint ) );
			}
			else
			{
				memcpy( world->joints.data + index, p, sizeof( b3Joint ) );
			}
			return;

		case b3_dtIsland:
			b3DeltaApplyIsland( world, index, p, len );
			return;

		case b3_dtSet:
			b3DeltaApplySet( world, index, p, len );
			return;

		default:
			break;
	}

	if ( b3_dtPool0 <= table && table < b3_dtPool0 + b3_deCount )
	{
		b3IdPool* pool = b3DeltaPool( world, table - b3_dtPool0 );
		if ( index > 0 )
		{
			b3DeltaApplyBlock( pool->freeArray.data, sizeof( int ), B3_DELTA_POOL_BLOCK, index - 1, p, len );
			return;
		}
		pool->nextIndex = b3DeltaTakeI32( &p );
		int count = b3DeltaTakeI32( &p );
		b3Array_Resize( pool->freeArray, count );
		return;
	}
	if ( b3_dtTreeHeader0 <= table && table < b3_dtTreeHeader0 + b3_bodyTypeCount )
	{
		b3DeltaApplyTreeHeader( b3DeltaTree( world, table, b3_dtTreeHeader0 ), p );
		return;
	}
	if ( b3_dtTreeNodes0 <= table && table < b3_dtTreeNodes0 + b3_bodyTypeCount )
	{
		b3DynamicTree* tree = b3DeltaTree( world, table, b3_dtTreeNodes0 );
		b3DeltaApplyBlock( tree->nodes, sizeof( b3TreeNode ), B3_DELTA_NODE_BLOCK, index, p, len );
		return;
	}
	if ( b3_dtTreeParents0 <= table && table < b3_dtTreeParents0 + b3_bodyTypeCount )
	{
		b3DynamicTree* tree = b3DeltaTree( world, table, b3_dtTreeParents0 );
		b3DeltaApplyBlock( tree->parents, sizeof( int32_t ), B3_DELTA_PARENT_BLOCK, index, p, len );
		return;
	}
	if ( b3_dtTreeProxies0 <= table && table < b3_dtTreeProxies0 + b3_bodyTypeCount )
	{
		b3DynamicTree* tree = b3DeltaTree( world, table, b3_dtTreeProxies0 );
		b3DeltaApplyBlock( tree->proxies, sizeof( b3TreeProxy ), B3_DELTA_PROXY_BLOCK, index, p, len );
		return;
	}
	B3_ASSERT( false );
}

// Capture

static void b3DeltaDiffContainer( b3WorldDelta* d, int table, const uint8_t* oldBytes, int oldLen, const uint8_t* newBytes, int newLen );

// Compare one slot against the mirror, logging the old image if it changed
static bool b3DeltaCompare( b3WorldDelta* d, b3DeltaBuffer* log, int table, int index )
{
	int len;
	const uint8_t* image = b3DeltaImage( d, table, index, &len );
	b3DeltaMirror* m = d->mirrors + table;
	d->stats.comparedCount += 1;
	d->tableCompared[table] += 1;

	if ( index >= m->count && len == B3_DELTA_ABSENT )
	{
		return false;
	}

	b3DeltaBlob* blob = b3DeltaMirrorSlot( d, table, index );
	if ( blob->len == len && ( len <= 0 || memcmp( blob->data, image, len ) == 0 ) )
	{
		return false;
	}

	if ( d->verify )
	{
		int limit = b3MinInt( blob->len, len );
		int offset = 0;
		while ( offset < limit && blob->data[offset] == image[offset] )
		{
			offset += 1;
		}
		d->lastDiffOffset = offset;
	}

	int before = log->size;
	b3DeltaAppendRecord( log, table, index, blob->data, blob->len );
	d->stats.logBytes += log->size - before;
	d->stats.changedCount += 1;
	d->stats.changedBytes += log->size - before;
	d->tableBytes[table] += log->size - before;

	b3DeltaDiffContainer( d, table, blob->data, blob->len, image, len );

	// The old image's hull reference moves to the log, the new image takes its own
	b3DeltaBlobSet( d, blob, image, len );
	b3DeltaRetain( d, table, index, image, len );
	return true;
}

static void b3DeltaCompareAll( b3WorldDelta* d, b3DeltaBuffer* log, int table )
{
	int count = b3MaxInt( b3DeltaSlotCount( d->world, table ), d->mirrors[table].count );
	for ( int i = 0; i < count; ++i )
	{
		b3DeltaCompare( d, log, table, i );
	}
}

// Pushes onto marks (list 0) or markRoots (list 1) once per capture
static void b3DeltaPushMark( b3WorldDelta* d, int list, int element, int id )
{
	uint32_t** stamps = &d->markStamps[list][element];
	*stamps = b3DeltaGrowStamps( *stamps, &d->markStampCapacity[list][element], id + 1 );
	if ( ( *stamps )[id] != d->markEpoch )
	{
		( *stamps )[id] = d->markEpoch;
		b3DeltaPushId( list == 0 ? d->marks + element : d->markRoots + element, id );
	}
}

static void b3DeltaAdd( b3WorldDelta* d, int element, int id )
{
	if ( id < 0 )
	{
		return;
	}

	if ( d->markMode )
	{
		b3DeltaPushMark( d, 0, element, id );
		return;
	}

	d->stamps[element] = b3DeltaGrowStamps( d->stamps[element], d->stampCapacity + element, id + 1 );
	if ( d->stamps[element][id] != d->stampValue )
	{
		d->stamps[element][id] = d->stampValue;
		b3DeltaPushId( d->candidates + element, id );
	}
}

// Solver sets and islands keep member arrays with swap removal. The member stores its position
// (localIndex, islandIndex), so a member whose position changed was rewritten even when nothing
// else about it did. Diffing the arrays by position between the old and new image finds exactly
// those members.
typedef struct b3DeltaMemberArray
{
	int elementSize;
	int idOffset;
	int element;

	// A body's solver set position is cached in its contacts (encodedBodySim), so a moved body brings
	// its neighborhood
	bool expand;
} b3DeltaMemberArray;

static void b3DeltaExpandBody( b3WorldDelta* d, int bodyId, bool marked );

static void b3DeltaDiffMembers( b3WorldDelta* d, const uint8_t* oldBytes, int oldLen, const uint8_t* newBytes, int newLen,
								int headerSize, const b3DeltaMemberArray* arrays, int arrayCount )
{
	const uint8_t* po = oldLen > 0 ? oldBytes + headerSize : NULL;
	const uint8_t* pn = newLen > 0 ? newBytes + headerSize : NULL;
	for ( int a = 0; a < arrayCount; ++a )
	{
		int oldCount = po != NULL ? b3DeltaTakeI32( &po ) : 0;
		int newCount = pn != NULL ? b3DeltaTakeI32( &pn ) : 0;
		const b3DeltaMemberArray* array = arrays + a;
		if ( array->element >= 0 )
		{
			int count = b3MaxInt( oldCount, newCount );
			for ( int i = 0; i < count; ++i )
			{
				int oldId = B3_NULL_INDEX, newId = B3_NULL_INDEX;
				if ( i < oldCount )
				{
					memcpy( &oldId, po + i * array->elementSize + array->idOffset, sizeof( int ) );
				}
				if ( i < newCount )
				{
					memcpy( &newId, pn + i * array->elementSize + array->idOffset, sizeof( int ) );
				}
				if ( oldId != newId && array->expand )
				{
					b3DeltaExpandBody( d, oldId, false );
					b3DeltaExpandBody( d, newId, false );
				}
				else if ( oldId != newId )
				{
					b3DeltaAdd( d, array->element, oldId );
					b3DeltaAdd( d, array->element, newId );
				}
			}
		}
		if ( po != NULL )
		{
			po += oldCount * array->elementSize;
		}
		if ( pn != NULL )
		{
			pn += newCount * array->elementSize;
		}
	}
}

// Adds an id read from a slot image. Freed slots keep stale ids, so bound them by the table.
static void b3DeltaAddLink( b3WorldDelta* d, int element, int id )
{
	int table = b3_dtBody + element;
	if ( id < b3MaxInt( b3DeltaSlotCount( d->world, table ), d->mirrors[table].count ) )
	{
		b3DeltaAdd( d, element, id );
	}
}

static void b3DeltaAddLinks( b3WorldDelta* d, int table, const uint8_t* bytes, int len )
{
	if ( len <= 0 )
	{
		return;
	}

	if ( table == b3_dtShape )
	{
		b3Shape shape;
		memcpy( &shape, bytes, sizeof( b3Shape ) );
		b3DeltaAddLink( d, b3_deBody, shape.bodyId );
		b3DeltaAddLink( d, b3_deShape, shape.prevShapeId );
		b3DeltaAddLink( d, b3_deShape, shape.nextShapeId );
	}
	else if ( table == b3_dtContact )
	{
		b3Contact contact;
		memcpy( &contact, bytes, sizeof( b3Contact ) );
		b3DeltaAddLink( d, b3_deSet, contact.setIndex );
		b3DeltaAddLink( d, b3_deIsland, contact.islandId );
		for ( int i = 0; i < 2; ++i )
		{
			b3ContactEdge* edge = contact.edges + i;
			b3DeltaAddLink( d, b3_deBody, edge->bodyId );
			if ( edge->prevKey != B3_NULL_INDEX )
			{
				b3DeltaAddLink( d, b3_deContact, edge->prevKey >> 1 );
			}
			if ( edge->nextKey != B3_NULL_INDEX )
			{
				b3DeltaAddLink( d, b3_deContact, edge->nextKey >> 1 );
			}
		}
	}
	else
	{
		b3Joint joint;
		memcpy( &joint, bytes, sizeof( b3Joint ) );
		b3DeltaAddLink( d, b3_deSet, joint.setIndex );
		b3DeltaAddLink( d, b3_deIsland, joint.islandId );
		for ( int i = 0; i < 2; ++i )
		{
			b3JointEdge* edge = joint.edges + i;
			b3DeltaAddLink( d, b3_deBody, edge->bodyId );
			if ( edge->prevKey != B3_NULL_INDEX )
			{
				b3DeltaAddLink( d, b3_deJoint, edge->prevKey >> 1 );
			}
			if ( edge->nextKey != B3_NULL_INDEX )
			{
				b3DeltaAddLink( d, b3_deJoint, edge->nextKey >> 1 );
			}
		}
	}
}

// Sensor image: count, then per sensor its shapeId and three visitor arrays
static const uint8_t* b3DeltaSkipSensor( const uint8_t* p, int* shapeId )
{
	*shapeId = b3DeltaTakeI32( &p );
	for ( int i = 0; i < 3; ++i )
	{
		int count = b3DeltaTakeI32( &p );
		p += count * (int)sizeof( b3Visitor );
	}
	return p;
}

static void b3DeltaDiffSensors( b3WorldDelta* d, const uint8_t* oldBytes, int oldLen, const uint8_t* newBytes, int newLen )
{
	const uint8_t* po = oldBytes;
	const uint8_t* pn = newBytes;
	int oldCount = oldLen > 0 ? b3DeltaTakeI32( &po ) : 0;
	int newCount = newLen > 0 ? b3DeltaTakeI32( &pn ) : 0;
	int count = b3MaxInt( oldCount, newCount );
	for ( int i = 0; i < count; ++i )
	{
		int oldId = B3_NULL_INDEX, newId = B3_NULL_INDEX;
		if ( i < oldCount )
		{
			po = b3DeltaSkipSensor( po, &oldId );
		}
		if ( i < newCount )
		{
			pn = b3DeltaSkipSensor( pn, &newId );
		}
		if ( oldId != newId )
		{
			b3DeltaAddLink( d, b3_deShape, oldId );
			b3DeltaAddLink( d, b3_deShape, newId );
		}
	}
}

static void b3DeltaDiffContainer( b3WorldDelta* d, int table, const uint8_t* oldBytes, int oldLen, const uint8_t* newBytes, int newLen )
{
	if ( table == b3_dtSet )
	{
		static const b3DeltaMemberArray arrays[] = {
			{ sizeof( b3BodySim ), offsetof( b3BodySim, bodyId ), b3_deBody, true },
			{ sizeof( b3BodyState ), 0, -1, false },
			{ sizeof( b3JointSim ), offsetof( b3JointSim, jointId ), b3_deJoint, false },
			{ sizeof( int ), 0, b3_deContact, false },
			{ sizeof( b3IslandSim ), offsetof( b3IslandSim, islandId ), b3_deIsland, false },
		};
		b3DeltaDiffMembers( d, oldBytes, oldLen, newBytes, newLen, sizeof( int32_t ), arrays, 5 );
	}
	else if ( table == b3_dtIsland )
	{
		static const b3DeltaMemberArray arrays[] = {
			{ sizeof( int ), 0, b3_deBody, false },
			{ sizeof( b3ContactLink ), offsetof( b3ContactLink, contactId ), b3_deContact, false },
			{ sizeof( b3JointLink ), offsetof( b3JointLink, jointId ), b3_deJoint, false },
		};
		b3DeltaDiffMembers( d, oldBytes, oldLen, newBytes, newLen, 4 * sizeof( int32_t ), arrays, 3 );
	}
	else if ( table == b3_dtSensors )
	{
		// Removal swaps the last sensor into the hole and rewrites its shape's sensorIndex
		b3DeltaDiffSensors( d, oldBytes, oldLen, newBytes, newLen );
	}
	else if ( table == b3_dtShape || table == b3_dtContact || table == b3_dtJoint )
	{
		// Linking or unlinking a list member rewrites its neighbors and its owners. The old image names
		// them for an unlink, the new image for a link. This covers topology changes of bodies that were
		// in no neighborhood, like a contact of a body woken this tick ending during the step.
		b3DeltaAddLinks( d, table, oldBytes, oldLen );
		b3DeltaAddLinks( d, table, newBytes, newLen );
	}
}

static void b3DeltaExpandBody( b3WorldDelta* d, int bodyId, bool marked )
{
	b3World* world = d->world;
	if ( bodyId < 0 || bodyId >= world->bodies.count )
	{
		return;
	}

	if ( d->markMode == false )
	{
		int kind = marked ? 1 : 0;
		d->expandStamps[kind] = b3DeltaGrowStamps( d->expandStamps[kind], d->expandCapacity + kind, bodyId + 1 );
		bool done = d->expandStamps[kind][bodyId] == d->stampValue;
		if ( marked == false && d->expandStamps[1] != NULL && bodyId < d->expandCapacity[1] )
		{
			// A marked expansion is a superset of a plain one
			done = done || d->expandStamps[1][bodyId] == d->stampValue;
		}
		if ( done )
		{
			return;
		}
		d->expandStamps[kind][bodyId] = d->stampValue;
	}

	b3DeltaAdd( d, b3_deBody, bodyId );
	b3Body* body = world->bodies.data + bodyId;
	if ( body->id != bodyId )
	{
		return;
	}

	b3DeltaAdd( d, b3_deIsland, body->islandId );

	// A set freed by a wake and reallocated by a sleep in the same step leaves the id pool unchanged
	b3DeltaAdd( d, b3_deSet, body->setIndex );

	for ( int shapeId = body->headShapeId; shapeId != B3_NULL_INDEX; shapeId = world->shapes.data[shapeId].nextShapeId )
	{
		b3DeltaAdd( d, b3_deShape, shapeId );
	}

	int contactKey = body->headContactKey;
	while ( contactKey != B3_NULL_INDEX )
	{
		int contactId = contactKey >> 1;
		int edgeIndex = contactKey & 1;
		b3Contact* contact = world->contacts.data + contactId;
		b3ContactEdge* other = contact->edges + ( edgeIndex ^ 1 );
		b3DeltaAdd( d, b3_deContact, contactId );
		b3DeltaAdd( d, b3_deBody, other->bodyId );
		if ( other->prevKey != B3_NULL_INDEX )
		{
			b3DeltaAdd( d, b3_deContact, other->prevKey >> 1 );
		}
		if ( other->nextKey != B3_NULL_INDEX )
		{
			b3DeltaAdd( d, b3_deContact, other->nextKey >> 1 );
		}
		contactKey = contact->edges[edgeIndex].nextKey;
	}

	int jointKey = body->headJointKey;
	while ( jointKey != B3_NULL_INDEX )
	{
		int jointId = jointKey >> 1;
		int edgeIndex = jointKey & 1;
		b3Joint* joint = world->joints.data + jointId;
		b3JointEdge* other = joint->edges + ( edgeIndex ^ 1 );
		b3DeltaAdd( d, b3_deJoint, jointId );
		b3DeltaAdd( d, b3_deBody, other->bodyId );
		b3DeltaAdd( d, b3_deIsland, joint->islandId );
		if ( marked )
		{
			if ( other->prevKey != B3_NULL_INDEX )
			{
				b3DeltaAdd( d, b3_deJoint, other->prevKey >> 1 );
			}
			if ( other->nextKey != B3_NULL_INDEX )
			{
				b3DeltaAdd( d, b3_deJoint, other->nextKey >> 1 );
			}
			b3DeltaAdd( d, b3_deSet, joint->setIndex );
		}
		jointKey = joint->edges[edgeIndex].nextKey;
	}
}

static void b3DeltaExpandJoint( b3WorldDelta* d, int jointId )
{
	b3World* world = d->world;
	if ( jointId < 0 || jointId >= world->joints.count )
	{
		return;
	}
	b3DeltaAdd( d, b3_deJoint, jointId );
	b3Joint* joint = world->joints.data + jointId;
	if ( joint->jointId != jointId )
	{
		return;
	}
	b3DeltaAdd( d, b3_deIsland, joint->islandId );
	b3DeltaAdd( d, b3_deSet, joint->setIndex );
	for ( int i = 0; i < 2; ++i )
	{
		b3JointEdge* edge = joint->edges + i;
		if ( edge->prevKey != B3_NULL_INDEX )
		{
			b3DeltaAdd( d, b3_deJoint, edge->prevKey >> 1 );
		}
		if ( edge->nextKey != B3_NULL_INDEX )
		{
			b3DeltaAdd( d, b3_deJoint, edge->nextKey >> 1 );
		}
		b3DeltaExpandBody( d, edge->bodyId, true );
	}
}

static void b3DeltaExpandShape( b3WorldDelta* d, int shapeId )
{
	b3World* world = d->world;
	if ( shapeId < 0 || shapeId >= world->shapes.count )
	{
		return;
	}
	b3DeltaAdd( d, b3_deShape, shapeId );
	b3Shape* shape = world->shapes.data + shapeId;
	if ( shape->id == shapeId )
	{
		b3DeltaExpandBody( d, shape->bodyId, true );
	}
}

static void b3DeltaExpandRoot( b3WorldDelta* d, int element, int id )
{
	switch ( element )
	{
		case b3_deBody:
			b3DeltaExpandBody( d, id, true );
			break;
		case b3_deShape:
			b3DeltaExpandShape( d, id );
			break;
		case b3_deJoint:
			b3DeltaExpandJoint( d, id );
			break;
		case b3_deSet:
			b3DeltaAdd( d, b3_deSet, id );
			break;
		default:
			b3DeltaAdd( d, element, id );
			break;
	}
}

// Ids allocated or freed since the last capture: the pool's high water mark moved or the free
// stack diverged from the mirror.
static void b3DeltaAddPoolChanges( b3WorldDelta* d, int element )
{
	int table = b3_dtPool0 + element;
	b3IdPool* pool = b3DeltaPool( d->world, element );
	const uint8_t* p = b3DeltaMirrorSlot( d, table, 0 )->data;
	int oldNext = b3DeltaTakeI32( &p );
	int oldFreeCount = b3DeltaTakeI32( &p );

	int newNext = pool->nextIndex;
	for ( int id = b3MinInt( oldNext, newNext ); id < b3MaxInt( oldNext, newNext ); ++id )
	{
		b3DeltaExpandRoot( d, element, id );
	}

	// The free stack changes at its top, so skip equal leading blocks and then walk ids
	int newFreeCount = pool->freeArray.count;
	const int* newFree = pool->freeArray.data;
	int limit = b3MinInt( oldFreeCount, newFreeCount );
	int common = 0;
	int block = 0;
	while ( common + B3_DELTA_POOL_BLOCK <= limit )
	{
		const b3DeltaBlob* blob = b3DeltaMirrorSlot( d, table, 1 + block );
		if ( memcmp( blob->data, newFree + common, B3_DELTA_POOL_BLOCK * sizeof( int ) ) != 0 )
		{
			break;
		}
		common += B3_DELTA_POOL_BLOCK;
		block += 1;
	}

	for ( int i = common; i < b3MaxInt( oldFreeCount, newFreeCount ); ++i )
	{
		int oldId = B3_NULL_INDEX;
		if ( i < oldFreeCount )
		{
			const b3DeltaBlob* blob = b3DeltaMirrorSlot( d, table, 1 + i / B3_DELTA_POOL_BLOCK );
			memcpy( &oldId, blob->data + ( i % B3_DELTA_POOL_BLOCK ) * sizeof( int ), sizeof( int ) );
		}
		int newId = i < newFreeCount ? newFree[i] : B3_NULL_INDEX;
		if ( oldId != newId )
		{
			b3DeltaExpandRoot( d, element, oldId );
			b3DeltaExpandRoot( d, element, newId );
		}
	}
}

static void b3DeltaClearMarks( b3WorldDelta* d )
{
	for ( int i = 0; i < b3_deCount; ++i )
	{
		d->marks[i].count = 0;
		d->markRoots[i].count = 0;
	}
	d->markEpoch += 1;
	if ( d->markEpoch == 0 )
	{
		for ( int k = 0; k < 2; ++k )
		{
			for ( int i = 0; i < b3_deCount; ++i )
			{
				memset( d->markStamps[k][i], 0, d->markStampCapacity[k][i] * sizeof( uint32_t ) );
			}
		}
		d->markEpoch = 1;
	}
}

static void b3DeltaNextStamp( b3WorldDelta* d )
{
	d->stampValue += 1;
	if ( d->stampValue == 0 )
	{
		for ( int i = 0; i < b3_deCount; ++i )
		{
			memset( d->stamps[i], 0, d->stampCapacity[i] * sizeof( uint32_t ) );
		}
		for ( int k = 0; k < 2; ++k )
		{
			memset( d->expandStamps[k], 0, d->expandCapacity[k] * sizeof( uint32_t ) );
		}
		d->stampValue = 1;
	}
	for ( int i = 0; i < b3_deCount; ++i )
	{
		d->candidates[i].count = 0;
	}
}

// The neighborhood of the awake bodies, kept as the next capture's previous neighborhood
static void b3DeltaCollectAwakeNeighborhood( b3WorldDelta* d )
{
	b3SolverSet* awakeSet = d->world->solverSets.data + b3_awakeSet;
	for ( int i = 0; i < awakeSet->bodySims.count; ++i )
	{
		b3DeltaExpandBody( d, awakeSet->bodySims.data[i].bodyId, false );
	}
	for ( int i = 0; i < awakeSet->islandSims.count; ++i )
	{
		b3DeltaAdd( d, b3_deIsland, awakeSet->islandSims.data[i].islandId );
	}
}

static void b3DeltaSaveAwake( b3WorldDelta* d )
{
	b3SolverSet* awakeSet = d->world->solverSets.data + b3_awakeSet;
	d->previousAwake.count = 0;
	for ( int i = 0; i < awakeSet->bodySims.count; ++i )
	{
		b3DeltaPushId( &d->previousAwake, awakeSet->bodySims.data[i].bodyId );
	}
}

static void b3DeltaCaptureInto( b3WorldDelta* d, b3DeltaBuffer* log )
{
	b3World* world = d->world;

	d->stats.comparedCount = 0;
	d->stats.changedCount = 0;
	d->stats.changedBytes = 0;
	d->stats.missCount = 0;
	memset( d->tableBytes, 0, sizeof( d->tableBytes ) );
	memset( d->tableCompared, 0, sizeof( d->tableCompared ) );

	b3DeltaNextStamp( d );

	// Neighborhood of the bodies awake now, kept for the next capture. Elements destroyed next tick
	// are in it along with their current neighbors.
	b3DeltaCollectAwakeNeighborhood( d );
	for ( int i = 0; i < b3_deCount; ++i )
	{
		b3DeltaCopyIds( d->nextNeighborhood + i, d->candidates + i );
	}

	// Previous neighborhood: elements destroyed this tick and their former neighbors
	for ( int i = 0; i < b3_deCount; ++i )
	{
		b3DeltaIds* previous = d->previousNeighborhood + i;
		for ( int j = 0; j < previous->count; ++j )
		{
			b3DeltaAdd( d, i, previous->data[j] );
		}
	}

	// Bodies awake at the previous capture: new contacts of bodies that fell asleep this tick
	for ( int i = 0; i < d->previousAwake.count; ++i )
	{
		b3DeltaExpandBody( d, d->previousAwake.data[i], false );
	}

	// API marks: the eager pre-mutation neighborhood and the roots again against current topology
	for ( int i = 0; i < b3_deCount; ++i )
	{
		for ( int j = 0; j < d->marks[i].count; ++j )
		{
			if ( i == b3_deSet )
			{
				b3DeltaAdd( d, b3_deSet, d->marks[i].data[j] );
			}
			else
			{
				b3DeltaAdd( d, i, d->marks[i].data[j] );
			}
		}
		for ( int j = 0; j < d->markRoots[i].count; ++j )
		{
			b3DeltaExpandRoot( d, i, d->markRoots[i].data[j] );
		}
	}

	for ( int i = 0; i < b3_deCount; ++i )
	{
		b3DeltaAddPoolChanges( d, i );
	}

	// Sets always compared: awake (every step), disabled (wake moves contacts out of it)
	b3DeltaAdd( d, b3_deSet, b3_awakeSet );
	b3DeltaAdd( d, b3_deSet, b3_disabledSet );

	// Singletons and flat regions
	b3DeltaCompare( d, log, b3_dtWorld, 0 );
	b3DeltaCompare( d, log, b3_dtCounts, 0 );
	b3DeltaCompare( d, log, b3_dtNames, 0 );
	b3DeltaCompare( d, log, b3_dtSensors, 0 );
	b3DeltaCompareAll( d, log, b3_dtColors );
	b3DeltaCompare( d, log, b3_dtPairHeader, 0 );
	b3DeltaCompareAll( d, log, b3_dtPairItems );
	b3DeltaCompare( d, log, b3_dtEvents, 0 );
	for ( int i = 0; i < b3_deCount; ++i )
	{
		b3DeltaCompareAll( d, log, b3_dtPool0 + i );
	}
	for ( int t = 0; t < b3_bodyTypeCount; ++t )
	{
		b3DeltaCompare( d, log, b3_dtTreeHeader0 + t, 0 );
		b3DeltaCompareAll( d, log, b3_dtTreeNodes0 + t );
		b3DeltaCompareAll( d, log, b3_dtTreeParents0 + t );
		b3DeltaCompareAll( d, log, b3_dtTreeProxies0 + t );
	}

	// Candidate elements. A changed set or island adds its moved members, a changed shape, contact or
	// joint adds its list neighbors and owners, so compare until no table grows. Containers go first so
	// most additions land before their table's pass.
	static const int order[b3_deCount] = { b3_deSet, b3_deIsland, b3_deBody, b3_deShape, b3_deContact, b3_deJoint };
	int cursors[b3_deCount] = { 0 };
	for ( ;; )
	{
		bool progress = true;
		while ( progress )
		{
			progress = false;
			for ( int k = 0; k < b3_deCount; ++k )
			{
				int i = order[k];
				b3DeltaIds* ids = d->candidates + i;
				while ( cursors[i] < ids->count )
				{
					b3DeltaCompare( d, log, b3_dtBody + i, ids->data[cursors[i]] );
					cursors[i] += 1;
					progress = true;
				}
			}
		}

		if ( d->verify == false )
		{
			break;
		}

		// A missed slot's diff queues further candidates, the next round compares them
		int missCount = d->stats.missCount;
		for ( int i = 0; i < b3_deCount; ++i )
		{
			int table = b3_dtBody + i;
			int count = b3MaxInt( b3DeltaSlotCount( world, table ), d->mirrors[table].count );
			d->stamps[i] = b3DeltaGrowStamps( d->stamps[i], d->stampCapacity + i, count );
			for ( int j = 0; j < count; ++j )
			{
				if ( d->stamps[i][j] == d->stampValue )
				{
					continue;
				}
				if ( b3DeltaCompare( d, log, table, j ) )
				{
					if ( d->stats.missCount < B3_DELTA_MAX_MISS_LOGS )
					{
						static const char* names[b3_deCount] = { "body", "shape", "contact", "joint", "island", "set" };
						b3Log( "b3WorldDelta: missed %s %d at step %" PRIu64 " first diff at byte %d", names[i], j,
							   world->stepIndex, d->lastDiffOffset );
					}
					d->stats.missCount += 1;
				}
			}
		}

		if ( d->stats.missCount == missCount )
		{
			break;
		}
	}

	for ( int i = 0; i < b3_deCount; ++i )
	{
		b3DeltaIds swap = d->previousNeighborhood[i];
		d->previousNeighborhood[i] = d->nextNeighborhood[i];
		d->nextNeighborhood[i] = swap;
	}
	b3DeltaSaveAwake( d );
	b3DeltaClearMarks( d );
}

// Marks

static b3WorldDelta* b3DeltaBeginMark( b3World* world )
{
	b3WorldDelta* d = world->delta;
	if ( d == NULL || d->world == NULL || world->stepping )
	{
		return NULL;
	}
	d->markMode = true;
	return d;
}

void b3DeltaMarkBody( b3World* world, int bodyId )
{
	b3WorldDelta* d = b3DeltaBeginMark( world );
	if ( d == NULL )
	{
		return;
	}

	// Expand on every call: an earlier call's expansion predates mutations made since, like a joint
	// created on this body that the coming mutation destroys again
	b3DeltaPushMark( d, 1, b3_deBody, bodyId );
	b3DeltaExpandBody( d, bodyId, true );
	d->markMode = false;
}

void b3DeltaMarkShape( b3World* world, int shapeId )
{
	b3WorldDelta* d = b3DeltaBeginMark( world );
	if ( d == NULL )
	{
		return;
	}
	b3DeltaPushMark( d, 1, b3_deShape, shapeId );
	b3DeltaAdd( d, b3_deShape, shapeId );
	b3Shape* shape = world->shapes.data + shapeId;
	d->markMode = false;
	b3DeltaMarkBody( world, shape->bodyId );
}

void b3DeltaMarkJoint( b3World* world, int jointId )
{
	b3WorldDelta* d = b3DeltaBeginMark( world );
	if ( d == NULL )
	{
		return;
	}
	b3DeltaPushMark( d, 1, b3_deJoint, jointId );
	b3Joint* joint = world->joints.data + jointId;
	b3DeltaAdd( d, b3_deJoint, jointId );
	b3DeltaAdd( d, b3_deIsland, joint->islandId );
	b3DeltaAdd( d, b3_deSet, joint->setIndex );
	for ( int i = 0; i < 2; ++i )
	{
		b3JointEdge* edge = joint->edges + i;
		if ( edge->prevKey != B3_NULL_INDEX )
		{
			b3DeltaAdd( d, b3_deJoint, edge->prevKey >> 1 );
		}
		if ( edge->nextKey != B3_NULL_INDEX )
		{
			b3DeltaAdd( d, b3_deJoint, edge->nextKey >> 1 );
		}
	}
	d->markMode = false;
	b3DeltaMarkBody( world, joint->edges[0].bodyId );
	b3DeltaMarkBody( world, joint->edges[1].bodyId );
}

// Restore

static void b3DeltaGrowElementTables( b3World* world, const b3DeltaCounts* counts )
{
#define B3_GROW_TABLE( arr, n )                                                                                                  \
	do                                                                                                                           \
	{                                                                                                                            \
		int oldCount = ( arr ).count;                                                                                            \
		if ( ( n ) > oldCount )                                                                                                  \
		{                                                                                                                        \
			b3Array_Resize( arr, n );                                                                                            \
			memset( ( arr ).data + oldCount, 0, ( ( n ) - oldCount ) * sizeof( *( arr ).data ) );                                \
		}                                                                                                                        \
	}                                                                                                                            \
	while ( 0 )

	B3_GROW_TABLE( world->bodies, counts->counts[b3_deBody] );
	B3_GROW_TABLE( world->shapes, counts->counts[b3_deShape] );
	B3_GROW_TABLE( world->fatAABBs, counts->counts[b3_deShape] );
	B3_GROW_TABLE( world->contacts, counts->counts[b3_deContact] );
	B3_GROW_TABLE( world->joints, counts->counts[b3_deJoint] );
	B3_GROW_TABLE( world->islands, counts->counts[b3_deIsland] );
	B3_GROW_TABLE( world->solverSets, counts->counts[b3_deSet] );

#undef B3_GROW_TABLE
}

static void b3DeltaTruncateElementTables( b3World* world, const b3DeltaCounts* counts )
{
	// Slots above the target count were restored to absent, their heap is released
	world->bodies.count = counts->counts[b3_deBody];
	world->shapes.count = counts->counts[b3_deShape];
	world->fatAABBs.count = counts->counts[b3_deShape];
	world->contacts.count = counts->counts[b3_deContact];
	world->joints.count = counts->counts[b3_deJoint];
	world->islands.count = counts->counts[b3_deIsland];
	world->solverSets.count = counts->counts[b3_deSet];
}

static bool b3DeltaIsHeader( int table, int index )
{
	if ( b3_dtPool0 <= table && table < b3_dtPool0 + b3_deCount )
	{
		return index == 0;
	}
	return table == b3_dtPairHeader || ( b3_dtTreeHeader0 <= table && table < b3_dtTreeHeader0 + b3_bodyTypeCount );
}

// Apply one log to the world and the mirror. Headers first so blocks fit their arrays.
static void b3DeltaUndoLog( b3WorldDelta* d, b3DeltaBuffer* log )
{
	for ( int pass = 0; pass < 2; ++pass )
	{
		const uint8_t* p = log->data;
		const uint8_t* end = log->data + log->size;
		while ( p < end )
		{
			b3DeltaRecord record;
			const uint8_t* next = b3DeltaNextRecord( p, &record );
			if ( b3DeltaIsHeader( record.table, record.index ) == ( pass == 0 ) )
			{
				const uint8_t* bytes = p + sizeof( b3DeltaRecord );
				b3DeltaApply( d, record.table, record.index, bytes, record.len );

				// The mirror takes the log's image and its hull reference, the dropped image releases its own
				b3DeltaBlob* blob = b3DeltaMirrorSlot( d, record.table, record.index );
				b3DeltaRelease( d, record.table, record.index, blob->data, blob->len );
				b3DeltaBlobSet( d, blob, bytes, record.len );
			}
			p = next;
		}
	}
	d->stats.logBytes -= log->size;
	log->size = 0;
}

static void b3DeltaScanCounts( const b3DeltaBuffer* log, b3DeltaCounts* target, b3DeltaCounts* maximum )
{
	const uint8_t* p = log->data;
	const uint8_t* end = log->data + log->size;
	while ( p < end )
	{
		b3DeltaRecord record;
		const uint8_t* next = b3DeltaNextRecord( p, &record );
		if ( record.table == b3_dtCounts )
		{
			memcpy( target, p + sizeof( b3DeltaRecord ), sizeof( b3DeltaCounts ) );
			for ( int i = 0; i < b3_deCount; ++i )
			{
				maximum->counts[i] = b3MaxInt( maximum->counts[i], target->counts[i] );
			}
		}
		p = next;
	}
}

static void b3DeltaTrimMirror( b3WorldDelta* d )
{
	// Trailing absent slots are dropped so full scans stay bounded by the live table size
	for ( int table = 0; table < b3_dtCount; ++table )
	{
		b3DeltaMirror* m = d->mirrors + table;
		while ( m->count > 0 && m->slots[m->count - 1].len == B3_DELTA_ABSENT )
		{
			m->count -= 1;
		}
	}
}

// Public API

b3WorldDelta* b3CreateWorldDelta( b3WorldId worldId, int maxTicks )
{
	b3World* world = b3GetUnlockedWorldFromId( worldId );
	B3_ASSERT( world == NULL || world->delta == NULL );
	B3_ASSERT( maxTicks > 0 );
	if ( world == NULL || world->delta != NULL || maxTicks <= 0 )
	{
		return NULL;
	}

	b3WorldDelta* d = b3AllocZero( sizeof( b3WorldDelta ) );
	d->world = world;
	d->maxTicks = maxTicks;
	d->logs = b3AllocZero( maxTicks * sizeof( b3DeltaBuffer ) );
	d->stampValue = 0;
	d->markEpoch = 1;

	// Tick 0: fill the mirror with every slot
	b3DeltaBuffer discard = { 0 };
	for ( int table = 0; table < b3_dtCount; ++table )
	{
		b3DeltaCompareAll( d, &discard, table );
	}
	b3DeltaFreeBuffer( &discard );
	d->stats.logBytes = 0;

	b3DeltaNextStamp( d );
	b3DeltaCollectAwakeNeighborhood( d );
	for ( int i = 0; i < b3_deCount; ++i )
	{
		b3DeltaCopyIds( d->previousNeighborhood + i, d->candidates + i );
	}
	b3DeltaSaveAwake( d );

	world->delta = d;
	return d;
}

void b3DeltaDetachWorld( b3World* world )
{
	b3WorldDelta* d = world->delta;
	if ( d == NULL )
	{
		return;
	}

	for ( int i = 0; i < d->maxTicks; ++i )
	{
		b3DeltaDropLog( d, d->logs + i );
	}
	b3DeltaDropLog( d, &d->pending );

	b3DeltaMirror* shapes = d->mirrors + b3_dtShape;
	for ( int i = 0; i < shapes->count; ++i )
	{
		b3DeltaRelease( d, b3_dtShape, i, shapes->slots[i].data, shapes->slots[i].len );
	}

	world->delta = NULL;
	d->world = NULL;
}

void b3DestroyWorldDelta( b3WorldDelta* d )
{
	if ( d == NULL )
	{
		return;
	}

	if ( d->world != NULL )
	{
		b3DeltaDetachWorld( d->world );
	}

	for ( int table = 0; table < b3_dtCount; ++table )
	{
		b3DeltaMirror* m = d->mirrors + table;
		for ( int i = 0; i < m->capacity; ++i )
		{
			b3Free( m->slots[i].data, m->slots[i].capacity );
		}
		b3Free( m->slots, m->capacity * sizeof( b3DeltaBlob ) );
	}

	for ( int i = 0; i < d->maxTicks; ++i )
	{
		b3DeltaFreeBuffer( d->logs + i );
	}
	b3Free( d->logs, d->maxTicks * sizeof( b3DeltaBuffer ) );
	b3DeltaFreeBuffer( &d->pending );
	b3DeltaFreeBuffer( &d->scratch );
	for ( int k = 0; k < 2; ++k )
	{
		b3Free( d->expandStamps[k], d->expandCapacity[k] * sizeof( uint32_t ) );
	}

	for ( int i = 0; i < b3_deCount; ++i )
	{
		b3Free( d->stamps[i], d->stampCapacity[i] * sizeof( uint32_t ) );
		b3DeltaFreeIds( d->candidates + i );
		b3DeltaFreeIds( d->previousNeighborhood + i );
		b3DeltaFreeIds( d->nextNeighborhood + i );
		b3DeltaFreeIds( d->marks + i );
		b3DeltaFreeIds( d->markRoots + i );
	}
	b3DeltaFreeIds( &d->previousAwake );
	for ( int k = 0; k < 2; ++k )
	{
		for ( int i = 0; i < b3_deCount; ++i )
		{
			b3Free( d->markStamps[k][i], d->markStampCapacity[k][i] * sizeof( uint32_t ) );
		}
	}

	b3Free( d, sizeof( b3WorldDelta ) );
}

int b3WorldDelta_Capture( b3WorldDelta* d )
{
	// Refused while detached or inside a step, where the tables are changing under the workers
	B3_ASSERT( d->world != NULL && d->world->locked == false );
	if ( d->world == NULL || d->world->locked )
	{
		return B3_NULL_INDEX;
	}

	int tick = d->newestTick + 1;
	if ( tick - d->oldestTick > d->maxTicks )
	{
		d->oldestTick += 1;
		b3DeltaDropLog( d, d->logs + d->oldestTick % d->maxTicks );
	}

	b3DeltaBuffer* log = d->logs + tick % d->maxTicks;
	B3_ASSERT( log->size == 0 );
	b3DeltaCaptureInto( d, log );
	d->newestTick = tick;
	return tick;
}

bool b3WorldDelta_Restore( b3WorldDelta* d, int tick )
{
	b3World* world = d->world;
	B3_ASSERT( world != NULL && world->locked == false );
	if ( world == NULL || world->locked || tick < d->oldestTick || tick > d->newestTick )
	{
		return false;
	}

	// A recording has no rewind operation, so a replay would diverge after this point
	if ( world->recording != NULL )
	{
		b3Log( "b3WorldDelta: restore refused while recording" );
		return false;
	}

	// Changes made since the newest capture go first
	b3DeltaCaptureInto( d, &d->pending );

	b3DeltaCounts target;
	{
		int len;
		const uint8_t* current = b3DeltaImage( d, b3_dtCounts, 0, &len );
		memcpy( &target, current, sizeof( target ) );
	}
	b3DeltaCounts maximum = target;
	b3DeltaScanCounts( &d->pending, &target, &maximum );
	for ( int t = d->newestTick; t > tick; --t )
	{
		b3DeltaScanCounts( d->logs + t % d->maxTicks, &target, &maximum );
	}

	b3DeltaGrowElementTables( world, &maximum );

	b3DeltaUndoLog( d, &d->pending );
	for ( int t = d->newestTick; t > tick; --t )
	{
		b3DeltaUndoLog( d, d->logs + t % d->maxTicks );
	}

	b3DeltaTruncateElementTables( world, &target );
	b3DeltaTrimMirror( d );
	d->newestTick = tick;

	b3DeltaNextStamp( d );
	b3DeltaCollectAwakeNeighborhood( d );
	for ( int i = 0; i < b3_deCount; ++i )
	{
		b3DeltaCopyIds( d->previousNeighborhood + i, d->candidates + i );
	}
	b3DeltaSaveAwake( d );
	b3DeltaClearMarks( d );
	return true;
}

int b3WorldDelta_GetNewestTick( const b3WorldDelta* d )
{
	return d->newestTick;
}

int b3WorldDelta_GetOldestTick( const b3WorldDelta* d )
{
	return d->oldestTick;
}

void b3WorldDelta_EnableVerify( b3WorldDelta* d, bool flag )
{
	d->verify = flag;
}

b3WorldDeltaStats b3WorldDelta_GetStats( const b3WorldDelta* d )
{
	return d->stats;
}

void b3DeltaPrintTableBytes( const b3WorldDelta* d )
{
	static const char* names[b3_dtCount] = {
		"world",	 "counts",	   "names",		 "sensors",		"colors",	   "pairHeader", "pairItems", "moveEvents",
		"poolBody",	 "poolShape",  "poolContact", "poolJoint",	"poolIsland",  "poolSet",	 "treeHdr0",  "treeHdr1",
		"treeHdr2",	 "treeNodes0", "treeNodes1", "treeNodes2", "treeParents0", "treeParents1", "treeParents2",
		"treeProxies0", "treeProxies1", "treeProxies2", "body", "shape", "contact", "joint", "island", "set",
	};
	for ( int i = 0; i < b3_dtCount; ++i )
	{
		if ( d->tableBytes[i] > 0 || d->tableCompared[i] > 0 )
		{
			b3Log( "  %-13s %8d bytes %7d compared", names[i], d->tableBytes[i], d->tableCompared[i] );
		}
	}
}
