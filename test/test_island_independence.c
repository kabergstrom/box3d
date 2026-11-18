// SPDX-License-Identifier: MIT

// Island independence: does an island step the same, bit for bit, whatever else the world
// holds? A pile alone, and the same pile beside a second pile far away (created before or
// after it), stepped together. The first step where the pile's transforms differ, if any, is
// printed: local rollback (restore and step one island) can only be deterministic where the
// answer is "never".

#include "test_macros.h"

#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "box3d/math_functions.h"

#include <stdio.h>
#include <string.h>

#define PILE_COUNT 40
#define STEP_COUNT 360

static bool s_enableSleep = true;

typedef struct Pile
{
	b3BodyId bodies[PILE_COUNT];
} Pile;

// A jumbled stack of boxes over x = cx, some rotated, so contacts start, stack and slide.
static void CreatePile( b3WorldId worldId, float cx, bool awake, Pile* pile )
{
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	for ( int i = 0; i < PILE_COUNT; ++i )
	{
		int layer = i / 4;
		int slot = i % 4;
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.isAwake = awake;
		bodyDef.position = ( b3Pos ){ cx + 1.1f * (float)( slot % 2 ) + 0.05f * (float)layer, 0.6f + 1.05f * (float)layer,
									  1.1f * (float)( slot / 2 ) - 0.03f * (float)layer };
		bodyDef.rotation = b3MakeQuatFromAxisAngle( b3Normalize( ( b3Vec3 ){ 0.3f, 1.0f, 0.2f * (float)slot } ), 0.15f * (float)i );
		pile->bodies[i] = b3CreateBody( worldId, &bodyDef );
		b3BoxHull box = b3MakeBoxHull( 0.5f, 0.5f, 0.5f );
		b3CreateHullShape( pile->bodies[i], &shapeDef, &box.base );
	}
}

static void CreateGround( b3WorldId worldId )
{
	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3BodyId ground = b3CreateBody( worldId, &bodyDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3BoxHull box = b3MakeBoxHull( 200.0f, 1.0f, 200.0f );
	box.base.center.y -= 1.0f;
	b3CreateHullShape( ground, &shapeDef, &box.base );
}

typedef enum Layout
{
	e_alone,
	e_otherBefore,
	e_otherAfter,
	// As e_otherBefore and e_otherAfter, the other pile asleep from the start (parked).
	e_otherBeforeAsleep,
	e_otherAfterAsleep,
} Layout;

// Pile A's transforms each step, with the world laid out as `layout`.
static void Run( Layout layout, int workerCount, b3WorldTransform* out );
static void CreatePile( b3WorldId worldId, float cx, bool awake, Pile* pile );
static void Run( Layout layout, int workerCount, b3WorldTransform* out )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.workerCount = workerCount;
	worldDef.enableSleep = s_enableSleep;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	CreateGround( worldId );
	Pile a, b;
	bool asleep = layout == e_otherBeforeAsleep || layout == e_otherAfterAsleep;
	if ( layout == e_otherBefore || layout == e_otherBeforeAsleep )
	{
		CreatePile( worldId, 60.0f, asleep == false, &b );
	}
	CreatePile( worldId, 0.0f, true, &a );
	if ( layout == e_otherAfter || layout == e_otherAfterAsleep )
	{
		CreatePile( worldId, 60.0f, asleep == false, &b );
	}
	for ( int step = 0; step < STEP_COUNT; ++step )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
		for ( int i = 0; i < PILE_COUNT; ++i )
		{
			out[step * PILE_COUNT + i] = b3Body_GetTransform( a.bodies[i] );
		}
	}
	b3DestroyWorld( worldId );
}

// The first step whose pile transforms differ, or -1.
static int FirstDifference( const b3WorldTransform* x, const b3WorldTransform* y )
{
	for ( int step = 0; step < STEP_COUNT; ++step )
	{
		if ( memcmp( x + step * PILE_COUNT, y + step * PILE_COUNT, PILE_COUNT * sizeof( b3WorldTransform ) ) != 0 )
		{
			return step;
		}
	}
	return -1;
}

static b3WorldTransform s_alone[STEP_COUNT * PILE_COUNT];
static b3WorldTransform s_other[STEP_COUNT * PILE_COUNT];
static b3WorldTransform s_parked[STEP_COUNT * PILE_COUNT];

static int Independence( void );

int IslandIndependenceTest( void )
{
	int failures = 0;
	for ( int sleep = 1; sleep >= 0; --sleep )
	{
		s_enableSleep = sleep == 1;
		printf( "  sleep %s\n", s_enableSleep ? "on" : "off" );
		failures += Independence();
	}
	ENSURE( failures == 0 );
	return 0;
}

static int Independence( void )
{
	const char* names[] = { "alone", "other pile created before", "other pile created after" };
	int failures = 0;
	for ( int workerCount = 1; workerCount <= 4; workerCount += 3 )
	{
		Run( e_alone, workerCount, s_alone );
		// The scene must move: a pile that never touches anything proves nothing.
		ENSURE( memcmp( s_alone, s_alone + ( STEP_COUNT - 1 ) * PILE_COUNT, PILE_COUNT * sizeof( b3WorldTransform ) ) != 0 );
		for ( int layout = e_otherBefore; layout <= e_otherAfter; ++layout )
		{
			Run( (Layout)layout, workerCount, s_other );
			int step = FirstDifference( s_alone, s_other );
			printf( "  island independence: workers %d, %s: %s", workerCount, names[layout], step < 0 ? "identical\n" : "" );
			if ( step >= 0 )
			{
				printf( "first difference at step %d\n", step );
				failures += 1;
			}
		}
	}
	// The question local rollback asks: in one world (ids as they are), does parking the other
	// pile (asleep, not stepped) change this pile's steps?
	for ( int workerCount = 1; workerCount <= 4; workerCount += 3 )
	{
		for ( int layout = e_otherBefore; layout <= e_otherAfter; ++layout )
		{
			Run( (Layout)layout, workerCount, s_other );
			Run( (Layout)( layout + 2 ), workerCount, s_parked );
			int step = FirstDifference( s_other, s_parked );
			printf( "  parked vs awake: workers %d, %s: ", workerCount, names[layout] );
			if ( step < 0 )
			{
				printf( "identical\n" );
			}
			else
			{
				printf( "first difference at step %d\n", step );
				failures += 1;
			}
		}
	}
	return failures;
}
