// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "test_macros.h"

#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "box3d/math_functions.h"

// Reach into internals for the state hash, event arrays and sensor bookkeeping
#include "body.h"
#include "joint.h"
#include "physics_world.h"
#include "recording.h"
#include "sensor.h"
#include "shape.h"

#include <stdlib.h>
#include <string.h>

// Random API ops, step and capture every tick, and restore a random tick in the window every few
// ticks. The oracle is a per tick hash of body state, event arrays and world host pointers:
// - the restored world hashes equal to the original at that tick
// - replaying the same ops from there reproduces the original hashes
// - verification finds no change the candidate set missed

static uint32_t s_rng;

static uint32_t Rand( void )
{
	s_rng ^= s_rng << 13;
	s_rng ^= s_rng >> 17;
	s_rng ^= s_rng << 5;
	return s_rng;
}

static float RandRange( float a, float b )
{
	return a + ( b - a ) * ( Rand() >> 8 ) * ( 1.0f / 16777216.0f );
}

static int s_contexts[4] = { 0, 1, 2, 3 };

static bool FilterA( b3ShapeId a, b3ShapeId b, void* context )
{
	B3_UNUSED( context );
	return ( ( a.index1 + b.index1 ) % 3 ) != 0;
}

static bool FilterB( b3ShapeId a, b3ShapeId b, void* context )
{
	return ( ( a.index1 ^ b.index1 ) & 1 ) == ( *(int*)context & 1 );
}

static uint64_t HashBytes( uint64_t hash, const void* data, int size )
{
	const uint8_t* p = data;
	for ( int i = 0; i < size; ++i )
	{
		hash = ( hash ^ p[i] ) * 0x100000001b3ull;
	}
	return hash;
}

#define HASH_ARRAY( h, a ) h = HashBytes( HashBytes( h, &( a ).count, sizeof( int ) ), ( a ).data, ( a ).count * (int)sizeof( *( a ).data ) )

// Body state plus what b3HashWorldState leaves out and the delta must still restore
static uint64_t HashWorld( b3World* world )
{
	uint64_t h = b3HashWorldState( world );
	h = HashBytes( h, &world->bodies.count, sizeof( int ) );
	h = HashBytes( h, &world->contacts.count, sizeof( int ) );
	HASH_ARRAY( h, world->bodyMoveEvents );
	HASH_ARRAY( h, world->sensorBeginEvents );
	HASH_ARRAY( h, world->contactBeginEvents );
	for ( int i = 0; i < 2; ++i )
	{
		HASH_ARRAY( h, world->sensorEndEvents[i] );
		HASH_ARRAY( h, world->contactEndEvents[i] );
	}
	HASH_ARRAY( h, world->jointEvents );
	for ( int i = 0; i < world->sensors.count; ++i )
	{
		b3Sensor* sensor = world->sensors.data + i;
		h = HashBytes( h, &sensor->shapeId, sizeof( int ) );
		HASH_ARRAY( h, sensor->overlaps2 );
	}
	h = HashBytes( h, &world->customFilterFcn, sizeof( void* ) );
	h = HashBytes( h, &world->customFilterContext, sizeof( void* ) );
	h = HashBytes( h, &world->userData, sizeof( void* ) );
	return h;
}

static int RandomLiveBody( b3World* world, bool dynamicOnly )
{
	int n = world->bodies.count;
	for ( int tries = 0; tries < 16 && n > 1; ++tries )
	{
		int i = 1 + (int)( Rand() % ( n - 1 ) );
		b3Body* body = world->bodies.data + i;
		if ( body->id == i && ( dynamicOnly == false || body->type == b3_dynamicBody ) )
		{
			return i;
		}
	}
	return B3_NULL_INDEX;
}

static b3BodyId MakeBodyId( b3World* world, int index )
{
	return (b3BodyId){ index + 1, world->worldId, world->bodies.data[index].generation };
}

static b3ShapeId HeadShapeId( b3World* world, int bodyIndex )
{
	int shapeIndex = world->bodies.data[bodyIndex].headShapeId;
	if ( shapeIndex == B3_NULL_INDEX )
	{
		return b3_nullShapeId;
	}
	return (b3ShapeId){ shapeIndex + 1, world->worldId, world->shapes.data[shapeIndex].generation };
}

static void SpawnBody( b3WorldId worldId )
{
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = ( Rand() % 10 ) == 0 ? b3_staticBody : b3_dynamicBody;
	if ( ( Rand() % 20 ) == 0 )
	{
		bodyDef.type = b3_kinematicBody;
		bodyDef.linearVelocity = (b3Vec3){ RandRange( -1.0f, 1.0f ), 0.0f, RandRange( -1.0f, 1.0f ) };
	}
	bodyDef.position = (b3Pos){ RandRange( -15.0f, 15.0f ), RandRange( 1.0f, 12.0f ), RandRange( -15.0f, 15.0f ) };
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );

	if ( ( Rand() % 8 ) == 0 )
	{
		b3Body_SetName( bodyId, ( Rand() % 2 ) ? "crate" : "barrel" );
	}

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.enableSensorEvents = true;
	shapeDef.enableContactEvents = true;
	shapeDef.enableHitEvents = ( Rand() % 2 ) == 0;
	shapeDef.enableCustomFiltering = ( Rand() % 3 ) == 0;

	if ( ( Rand() % 6 ) == 0 )
	{
		b3ShapeDef sensorDef = shapeDef;
		sensorDef.isSensor = true;
		b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, RandRange( 1.0f, 2.5f ) };
		b3CreateSphereShape( bodyId, &sensorDef, &sphere );
	}

	int shapeCount = 1 + ( Rand() % 3 == 0 );
	for ( int i = 0; i < shapeCount; ++i )
	{
		b3Vec3 offset = { 0.6f * i, 0.0f, 0.0f };
		switch ( Rand() % 4 )
		{
			case 0:
			{
				b3Sphere sphere = { offset, RandRange( 0.2f, 0.6f ) };
				b3CreateSphereShape( bodyId, &shapeDef, &sphere );
				break;
			}
			case 1:
			{
				b3Capsule capsule = { offset, b3Add( offset, (b3Vec3){ 0.0f, 0.5f, 0.0f } ), 0.25f };
				b3CreateCapsuleShape( bodyId, &shapeDef, &capsule );
				break;
			}
			case 2:
			{
				// Unique hull content: the hull database frees it when the shape goes away
				b3Vec3 points[12];
				for ( int k = 0; k < 12; ++k )
				{
					points[k] = b3Add( offset, (b3Vec3){ RandRange( -0.5f, 0.5f ), RandRange( -0.5f, 0.5f ), RandRange( -0.5f, 0.5f ) } );
				}
				b3HullData* hull = b3CreateHull( points, 12, 32 );
				if ( hull != NULL )
				{
					b3CreateHullShape( bodyId, &shapeDef, hull );
					b3DestroyHull( hull );
				}
				break;
			}
			default:
			{
				float h = RandRange( 0.3f, 0.7f );
				b3BoxHull box = b3MakeOffsetBoxHull( h, h, h, offset );
				b3CreateHullShape( bodyId, &shapeDef, &box.base );
				break;
			}
		}
	}
}

// Ops are seeded by tick so a replay after a restore issues the same calls
static void DoOps( b3WorldId worldId, int tick, uint32_t seed )
{
	b3World* world = b3GetWorldFromId( worldId );
	s_rng = ( seed ^ ( (uint32_t)tick * 2654435761u ) ) | 1u;
	Rand();

	int opCount = Rand() % 4;
	for ( int k = 0; k < opCount; ++k )
	{
		int op = Rand() % 110;
		if ( op < 18 )
		{
			SpawnBody( worldId );
		}
		else if ( op < 30 )
		{
			int b = RandomLiveBody( world, false );
			if ( b != B3_NULL_INDEX )
			{
				b3DestroyBody( MakeBodyId( world, b ) );
			}
		}
		else if ( op < 42 )
		{
			int b = RandomLiveBody( world, true );
			if ( b != B3_NULL_INDEX )
			{
				b3Vec3 v = { RandRange( -5.0f, 5.0f ), RandRange( 0.0f, 8.0f ), RandRange( -5.0f, 5.0f ) };
				b3Body_SetLinearVelocity( MakeBodyId( world, b ), v );
			}
		}
		else if ( op < 50 )
		{
			// Teleport, often a sleeping body
			int b = RandomLiveBody( world, false );
			if ( b != B3_NULL_INDEX )
			{
				b3Pos p = { RandRange( -15.0f, 15.0f ), RandRange( 1.0f, 10.0f ), RandRange( -15.0f, 15.0f ) };
				b3Body_SetTransform( MakeBodyId( world, b ), p, b3Quat_identity );
			}
		}
		else if ( op < 60 )
		{
			int a = RandomLiveBody( world, false );
			int b = RandomLiveBody( world, true );
			if ( a != B3_NULL_INDEX && b != B3_NULL_INDEX && a != b )
			{
				b3JointDef base = b3DefaultDistanceJointDef().base;
				base.bodyIdA = MakeBodyId( world, a );
				base.bodyIdB = MakeBodyId( world, b );
				base.collideConnected = Rand() % 2;
				if ( Rand() % 2 )
				{
					b3DistanceJointDef jointDef = b3DefaultDistanceJointDef();
					jointDef.base = base;
					jointDef.length = RandRange( 1.0f, 4.0f );
					jointDef.enableSpring = Rand() % 2;
					jointDef.hertz = 2.0f;
					jointDef.dampingRatio = 0.5f;
					b3CreateDistanceJoint( worldId, &jointDef );
				}
				else
				{
					b3RevoluteJointDef jointDef = b3DefaultRevoluteJointDef();
					jointDef.base = base;
					jointDef.base.localFrameA.p = (b3Vec3){ 0.5f, 0.0f, 0.0f };
					b3CreateRevoluteJoint( worldId, &jointDef );
				}
			}
		}
		else if ( op < 67 )
		{
			int n = world->joints.count;
			int j = n > 0 ? (int)( Rand() % n ) : B3_NULL_INDEX;
			if ( j != B3_NULL_INDEX && world->joints.data[j].jointId == j )
			{
				b3JointId jointId = { j + 1, world->worldId, world->joints.data[j].generation };
				b3DestroyJoint( jointId, Rand() % 2 );
			}
		}
		else if ( op < 72 )
		{
			int b = RandomLiveBody( world, false );
			if ( b != B3_NULL_INDEX )
			{
				static const b3BodyType types[3] = { b3_staticBody, b3_kinematicBody, b3_dynamicBody };
				b3Body_SetType( MakeBodyId( world, b ), types[Rand() % 3] );
			}
		}
		else if ( op < 78 )
		{
			int b = RandomLiveBody( world, false );
			if ( b != B3_NULL_INDEX )
			{
				b3BodyId bodyId = MakeBodyId( world, b );
				if ( b3Body_IsEnabled( bodyId ) )
				{
					b3Body_Disable( bodyId );
				}
				else
				{
					b3Body_Enable( bodyId );
				}
			}
		}
		else if ( op < 84 )
		{
			int b = RandomLiveBody( world, false );
			b3ShapeId shapeId = b != B3_NULL_INDEX ? HeadShapeId( world, b ) : b3_nullShapeId;
			if ( B3_IS_NON_NULL( shapeId ) )
			{
				b3Shape_SetFriction( shapeId, RandRange( 0.1f, 1.0f ) );
			}
		}
		else if ( op < 90 )
		{
			// Destroys sensors too, which swap removes them from the sensor array
			int b = RandomLiveBody( world, false );
			b3ShapeId shapeId = b != B3_NULL_INDEX ? HeadShapeId( world, b ) : b3_nullShapeId;
			if ( B3_IS_NON_NULL( shapeId ) )
			{
				b3DestroyShape( shapeId, true );
			}
		}
		else if ( op < 95 )
		{
			int b = RandomLiveBody( world, true );
			if ( b != B3_NULL_INDEX )
			{
				b3Body_SetAwake( MakeBodyId( world, b ), Rand() % 2 );
			}
		}
		else if ( op < 100 )
		{
			int b = RandomLiveBody( world, true );
			if ( b != B3_NULL_INDEX )
			{
				b3Body_ApplyLinearImpulseToCenter( MakeBodyId( world, b ), (b3Vec3){ 0.0f, RandRange( 1.0f, 10.0f ), 0.0f }, true );
			}
		}
		else if ( op < 104 )
		{
			b3CustomFilterFcn* const filters[3] = { NULL, FilterA, FilterB };
			b3CustomFilterFcn* filter = filters[Rand() % 3];
			b3World_SetCustomFilterCallback( worldId, filter, s_contexts + Rand() % 4 );
		}
		else if ( op < 106 )
		{
			b3World_SetUserData( worldId, s_contexts + Rand() % 4 );
		}
		else
		{
			// Geometry change in place keeps the shape generation
			int b = RandomLiveBody( world, false );
			b3ShapeId shapeId = b != B3_NULL_INDEX ? HeadShapeId( world, b ) : b3_nullShapeId;
			if ( B3_IS_NON_NULL( shapeId ) )
			{
				b3ShapeType type = b3Shape_GetType( shapeId );
				if ( type == b3_sphereShape || type == b3_capsuleShape )
				{
					if ( Rand() % 2 )
					{
						b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, RandRange( 0.2f, 0.6f ) };
						b3Shape_SetSphere( shapeId, &sphere );
					}
					else
					{
						b3Capsule capsule = { { 0.0f, 0.0f, 0.0f }, { 0.0f, RandRange( 0.2f, 0.8f ), 0.0f }, 0.25f };
						b3Shape_SetCapsule( shapeId, &capsule );
					}
				}
			}
		}
	}
}

static int RandomOpsRun( uint32_t seed, int workerCount, int tickCount )
{
	int64_t baseBytes = b3GetByteCount();

	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = workerCount;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	b3World* world = b3GetWorldFromId( worldId );

	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.position = (b3Pos){ 0.0f, -1.0f, 0.0f };
		b3BodyId groundId = b3CreateBody( worldId, &bodyDef );
		b3BoxHull box = b3MakeBoxHull( 40.0f, 1.0f, 40.0f );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		b3CreateHullShape( groundId, &shapeDef, &box.base );
	}

	s_rng = seed | 1u;
	for ( int i = 0; i < 150; ++i )
	{
		SpawnBody( worldId );
	}
	for ( int i = 0; i < 60; ++i )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
	}

	int window = 24;
	b3WorldDelta* delta = b3CreateWorldDelta( worldId, window );
	ENSURE( delta != NULL );
	b3WorldDelta_EnableVerify( delta, true );

	uint64_t* hashes = calloc( tickCount + 1, sizeof( uint64_t ) );
	hashes[0] = HashWorld( world );

	uint32_t controlRng = seed * 7919u + 1u;
	int restoreCount = 0;
	for ( int i = 0; i < tickCount; ++i )
	{
		int tick = b3WorldDelta_GetNewestTick( delta ) + 1;
		DoOps( worldId, tick, seed );
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
		ENSURE( b3WorldDelta_Capture( delta ) == tick );
		ENSURE( b3WorldDelta_GetStats( delta ).missCount == 0 );

		// Ticks below the high water mark are replays and must reproduce the original timeline
		uint64_t hash = HashWorld( world );
		if ( hashes[tick] != 0 )
		{
			ENSURE( hash == hashes[tick] );
		}
		hashes[tick] = hash;

		controlRng = controlRng * 1664525u + 1013904223u;
		if ( ( controlRng >> 24 ) % 9 != 0 )
		{
			continue;
		}

		int oldest = b3WorldDelta_GetOldestTick( delta );
		int target = oldest + (int)( ( controlRng >> 8 ) % ( tick - oldest + 1 ) );

		// Sometimes leave uncaptured changes behind, the restore must undo them too
		if ( ( controlRng >> 4 ) % 3 == 0 )
		{
			DoOps( worldId, tick + 100000, seed );
		}

		ENSURE( b3WorldDelta_Restore( delta, target ) );
		ENSURE( HashWorld( world ) == hashes[target] );
		restoreCount += 1;
	}

	ENSURE( restoreCount > 0 );

	free( hashes );
	b3DestroyWorldDelta( delta );
	b3DestroyWorld( worldId );
	ENSURE( b3GetByteCount() == baseBytes );
	return 0;
}

static int RandomOps( void )
{
	ENSURE( RandomOpsRun( 1, 1, 400 ) == 0 );
	ENSURE( RandomOpsRun( 2, 4, 400 ) == 0 );
	ENSURE( RandomOpsRun( 3, 1, 400 ) == 0 );
	return 0;
}

static b3ShapeId CreateStaticSensor( b3WorldId worldId, float x )
{
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.position = (b3Pos){ x, 0.0f, 0.0f };
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.isSensor = true;
	shapeDef.enableSensorEvents = true;
	b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 1.0f };
	return b3CreateSphereShape( bodyId, &shapeDef, &sphere );
}

// Destroying a sensor swap removes it, rewriting the moved sensor's shape. Nothing near an awake
// body touches that shape, so only the sensor array diff can find it. Verification stays off because
// it would catch and repair the miss.
static int SensorSwapRemove( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );
	b3World* world = b3GetWorldFromId( worldId );

	b3ShapeId sensorA = CreateStaticSensor( worldId, -10.0f );
	b3ShapeId sensorB = CreateStaticSensor( worldId, 10.0f );
	b3World_Step( worldId, 1.0f / 60.0f, 4 );

	b3WorldDelta* delta = b3CreateWorldDelta( worldId, 8 );
	b3DestroyShape( sensorA, true );
	b3World_Step( worldId, 1.0f / 60.0f, 4 );
	b3WorldDelta_Capture( delta );
	ENSURE( b3WorldDelta_Restore( delta, 0 ) );

	ENSURE( world->sensors.count == 2 );
	for ( int i = 0; i < 2; ++i )
	{
		b3ShapeId shapeId = i == 0 ? sensorA : sensorB;
		b3Shape* shape = world->shapes.data + ( shapeId.index1 - 1 );
		ENSURE( shape->sensorIndex != B3_NULL_INDEX );
		ENSURE( world->sensors.data[shape->sensorIndex].shapeId == shapeId.index1 - 1 );
	}

	b3DestroyWorldDelta( delta );
	b3DestroyWorld( worldId );
	return 0;
}

// Destroying an overlapped sensor queues an end event for the next step. Restoring a tick before the
// destruction brings the sensor back with its overlap, so that end event must go too.
static int QueuedEndEvent( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3ShapeId sensorId = CreateStaticSensor( worldId, 0.0f );
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.enableSleep = false;
		b3BodyId visitorId = b3CreateBody( worldId, &bodyDef );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		shapeDef.enableSensorEvents = true;
		b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.25f };
		b3CreateSphereShape( visitorId, &shapeDef, &sphere );
	}

	b3World_Step( worldId, 1.0f / 60.0f, 4 );
	ENSURE( b3World_GetSensorEvents( worldId ).beginCount == 1 );

	b3WorldDelta* delta = b3CreateWorldDelta( worldId, 8 );
	b3DestroyShape( sensorId, true );
	ENSURE( b3WorldDelta_Restore( delta, 0 ) );

	b3World_Step( worldId, 1.0f / 60.0f, 4 );
	b3SensorEvents events = b3World_GetSensorEvents( worldId );
	ENSURE( events.beginCount == 0 );
	ENSURE( events.endCount == 0 );

	b3DestroyWorldDelta( delta );
	b3DestroyWorld( worldId );
	return 0;
}

// Restores are refused inside a step and while recording, since neither can express a rewind
static int RefusedRestores( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );
	b3World* world = b3GetWorldFromId( worldId );

	b3WorldDelta* delta = b3CreateWorldDelta( worldId, 8 );
	ENSURE( b3WorldDelta_Capture( delta ) == 1 );
	ENSURE( b3WorldDelta_Restore( delta, 2 ) == false );

	b3Recording* recording = b3CreateRecording( 0 );
	b3World_StartRecording( worldId, recording );
	ENSURE( world->recording != NULL );
	ENSURE( b3WorldDelta_Restore( delta, 0 ) == false );
	b3World_StopRecording( worldId );
	ENSURE( b3WorldDelta_Restore( delta, 0 ) );

	b3DestroyRecording( recording );
	b3DestroyWorldDelta( delta );
	b3DestroyWorld( worldId );
	return 0;
}

int WorldDeltaTest( void )
{
	RUN_SUBTEST( RandomOps );
	RUN_SUBTEST( SensorSwapRemove );
	RUN_SUBTEST( QueuedEndEvent );
	RUN_SUBTEST( RefusedRestores );
	return 0;
}
