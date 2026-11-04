// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#include "body.h"
#include "core.h"
#include "joint.h"
#include "physics_world.h"
#include "recording.h"
#include "solver.h"
#include "solver_set.h"

// needed for dll export
#include "box3d/box3d.h"

b3RopeData* b3GetRopeData( b3World* world, b3JointSim* base )
{
	B3_ASSERT( base->type == b3_ropeJoint );
	b3RopeData* rope = b3Array_Get( world->ropes, base->ropeJoint.ropeId );
	B3_ASSERT( rope->jointId == base->jointId );
	return rope;
}

static b3RopeData* b3RopeOf( b3JointId jointId, bool mark )
{
	b3World* world = b3GetWorld( jointId.world0 );
	b3JointSim* base = mark ? b3GetJointSimCheckType( jointId, b3_ropeJoint ) : b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	return b3GetRopeData( world, base );
}

void b3RopeJoint_SetLength( b3JointId jointId, float length )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetLength, jointId, length );
	b3RopeOf( jointId, true )->length = b3ClampFloat( length, B3_LINEAR_SLOP, B3_HUGE );
}

float b3RopeJoint_GetLength( b3JointId jointId )
{
	return b3RopeOf( jointId, false )->length;
}

void b3RopeJoint_SetStiffness( b3JointId jointId, float stiffness )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetStiffness, jointId, stiffness );
	b3RopeOf( jointId, true )->stiffness = b3MaxFloat( stiffness, 0.0f );
}

float b3RopeJoint_GetStiffness( b3JointId jointId )
{
	return b3RopeOf( jointId, false )->stiffness;
}

void b3RopeJoint_SetDampingRatio( b3JointId jointId, float dampingRatio )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetDampingRatio, jointId, dampingRatio );
	b3RopeOf( jointId, true )->dampingRatio = b3MaxFloat( dampingRatio, 0.0f );
}

float b3RopeJoint_GetDampingRatio( b3JointId jointId )
{
	return b3RopeOf( jointId, false )->dampingRatio;
}

void b3RopeJoint_SetBendStiffness( b3JointId jointId, float bendStiffness )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetBendStiffness, jointId, bendStiffness );
	b3RopeOf( jointId, true )->bendStiffness = b3MaxFloat( bendStiffness, 0.0f );
}

float b3RopeJoint_GetBendStiffness( b3JointId jointId )
{
	return b3RopeOf( jointId, false )->bendStiffness;
}

void b3RopeJoint_SetBendDamping( b3JointId jointId, float bendDamping )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetBendDamping, jointId, bendDamping );
	b3RopeOf( jointId, true )->bendDamping = b3MaxFloat( bendDamping, 0.0f );
}

float b3RopeJoint_GetBendDamping( b3JointId jointId )
{
	return b3RopeOf( jointId, false )->bendDamping;
}

void b3RopeJoint_SetRestDirections( b3JointId jointId, int segment, b3Vec3 start, b3Vec3 end )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetRestDirections, jointId, segment, start, end );
	b3RopeData* rope = b3RopeOf( jointId, true );
	if ( segment < 0 || segment + 1 >= rope->pointCount || b3IsValidVec3( start ) == false || b3IsValidVec3( end ) == false )
	{
		return;
	}
	rope->restStart[segment] = start;
	rope->restEnd[segment] = end;
}

int b3RopeJoint_GetRestDirections( b3JointId jointId, b3Vec3* start, b3Vec3* end, int capacity )
{
	b3RopeData* rope = b3RopeOf( jointId, false );
	if ( start == NULL || end == NULL )
	{
		return 0;
	}
	int count = 0;
	for ( ; count + 1 < rope->pointCount && count < capacity; ++count )
	{
		start[count] = rope->restStart[count];
		end[count] = rope->restEnd[count];
	}
	return count;
}

// World points of the path, at most capacity.
static int b3RopeWorldPoints( b3World* world, const b3RopeData* rope, b3Pos* points, int capacity )
{
	b3WorldTransform transforms[B3_MAX_ROPE_BODIES];
	for ( int k = 0; k < rope->bodyCount; ++k )
	{
		transforms[k] = b3GetBodyTransform( world, rope->bodyIds[k] );
	}
	int count = rope->pointCount < capacity ? rope->pointCount : capacity;
	for ( int i = 0; i < count; ++i )
	{
		points[i] = b3TransformWorldPoint( transforms[rope->pointBodies[i]], rope->localPoints[i] );
	}
	return count;
}

float b3GetRopeJointLength( b3World* world, b3JointSim* base )
{
	b3Pos points[B3_MAX_ROPE_POINTS];
	int count = b3RopeWorldPoints( world, b3GetRopeData( world, base ), points, B3_MAX_ROPE_POINTS );
	float length = 0.0f;
	for ( int i = 0; i + 1 < count; ++i )
	{
		length += b3Length( b3SubPos( points[i + 1], points[i] ) );
	}
	return length;
}

void b3SetRopeJointRest( b3World* world, b3RopeData* rope )
{
	b3Pos points[B3_MAX_ROPE_POINTS];
	int count = b3RopeWorldPoints( world, rope, points, B3_MAX_ROPE_POINTS );
	b3Quat q[B3_MAX_ROPE_BODIES];
	for ( int k = 0; k < rope->bodyCount; ++k )
	{
		q[k] = b3GetBodyTransform( world, rope->bodyIds[k] ).q;
	}
	for ( int i = 0; i + 1 < count; ++i )
	{
		b3Vec3 d = b3SubPos( points[i + 1], points[i] );
		float length = b3Length( d );
		b3Vec3 u = length > FLT_EPSILON ? b3MulSV( 1.0f / length, d ) : b3Vec3_zero;
		rope->restStart[i] = b3InvRotateVector( q[rope->pointBodies[i]], u );
		rope->restEnd[i] = b3InvRotateVector( q[rope->pointBodies[i + 1]], u );
	}
}

float b3RopeJoint_GetCurrentLength( b3JointId jointId )
{
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	b3World* world = b3GetUnlockedWorld( jointId.world0 );
	if ( world == NULL )
	{
		return 0.0f;
	}
	return b3GetRopeJointLength( world, base );
}

// Segment i's tension impulse: the last segment's plus the friction impulses of the inner
// points after segment i.
static float b3RopeSegmentImpulse( const b3RopeData* rope, int segment )
{
	float impulse = rope->impulse;
	for ( int k = segment + 1; k + 1 < rope->pointCount; ++k )
	{
		impulse += rope->frictionImpulse[k];
	}
	return impulse;
}

float b3GetRopeJointMaxImpulse( const b3RopeData* rope )
{
	float most = 0.0f;
	for ( int i = 0; i + 1 < rope->pointCount; ++i )
	{
		most = b3MaxFloat( most, b3RopeSegmentImpulse( rope, i ) );
	}
	return most;
}

float b3RopeJoint_GetTension( b3JointId jointId )
{
	b3World* world = b3GetWorld( jointId.world0 );
	return world->inv_h * b3GetRopeJointMaxImpulse( b3RopeOf( jointId, false ) );
}

int b3RopeJoint_GetSegmentTensions( b3JointId jointId, float* tensions, int capacity )
{
	b3World* world = b3GetWorld( jointId.world0 );
	b3RopeData* rope = b3RopeOf( jointId, false );
	if ( tensions == NULL )
	{
		return 0;
	}
	int count = 0;
	for ( ; count + 1 < rope->pointCount && count < capacity; ++count )
	{
		tensions[count] = world->inv_h * b3RopeSegmentImpulse( rope, count );
	}
	return count;
}

int b3RopeJoint_GetSlipEnergies( b3JointId jointId, float* energies, int capacity )
{
	b3RopeData* rope = b3RopeOf( jointId, false );
	if ( energies == NULL )
	{
		return 0;
	}
	int count = rope->pointCount < capacity ? rope->pointCount : capacity;
	for ( int i = 0; i < count; ++i )
	{
		energies[i] = rope->slipEnergy[i];
	}
	return count;
}

int b3RopeJoint_GetPointCount( b3JointId jointId )
{
	return b3RopeOf( jointId, false )->pointCount;
}

int b3RopeJoint_GetBodies( b3JointId jointId, b3BodyId* bodies, int capacity )
{
	b3World* world = b3GetWorld( jointId.world0 );
	b3RopeData* rope = b3RopeOf( jointId, false );
	if ( bodies == NULL )
	{
		return 0;
	}
	int count = rope->bodyCount < capacity ? rope->bodyCount : capacity;
	for ( int k = 0; k < count; ++k )
	{
		bodies[k] = b3MakeBodyId( world, rope->bodyIds[k] );
	}
	return count;
}

int b3RopeJoint_GetPointBodies( b3JointId jointId, int* bodies, int capacity )
{
	b3RopeData* rope = b3RopeOf( jointId, false );
	if ( bodies == NULL )
	{
		return 0;
	}
	int count = rope->pointCount < capacity ? rope->pointCount : capacity;
	for ( int i = 0; i < count; ++i )
	{
		bodies[i] = rope->pointBodies[i];
	}
	return count;
}

int b3RopeJoint_GetWorldPoints( b3JointId jointId, b3Pos* points, int capacity )
{
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	b3World* world = b3GetUnlockedWorld( jointId.world0 );
	if ( world == NULL || points == NULL || capacity <= 0 )
	{
		return 0;
	}
	return b3RopeWorldPoints( world, b3GetRopeData( world, base ), points, capacity );
}

// The rope's pull on body 1 (the joint's body B): every segment pulls its ends toward each other.
b3Vec3 b3GetRopeJointForce( b3World* world, b3JointSim* base )
{
	b3RopeData* rope = b3GetRopeData( world, base );
	b3Pos points[B3_MAX_ROPE_POINTS];
	int count = b3RopeWorldPoints( world, rope, points, B3_MAX_ROPE_POINTS );
	b3Vec3 force = b3Vec3_zero;
	for ( int i = 0; i < count; ++i )
	{
		if ( rope->pointBodies[i] != 1 )
		{
			continue;
		}
		if ( i > 0 )
		{
			float tension = world->inv_h * b3RopeSegmentImpulse( rope, i - 1 );
			force = b3MulAdd( force, tension, b3Normalize( b3SubPos( points[i - 1], points[i] ) ) );
		}
		if ( i + 1 < count )
		{
			float tension = world->inv_h * b3RopeSegmentImpulse( rope, i );
			force = b3MulAdd( force, tension, b3Normalize( b3SubPos( points[i + 1], points[i] ) ) );
		}
	}
	return force;
}

// C = sum |p(i+1) - p(i)| - L <= 0
// u(i) = (p(i+1) - p(i)) / |p(i+1) - p(i)|
// Cdot = sum over points k of dot(g(k), v(k)), g(k) = u(k-1) - u(k) (missing terms zero)
// v(k) = v + cross(w, r(k)) of the point's body
// J = [Jv0 Jw0 ... Jv3 Jw3], JvX = sum g(k) over X's points, JwX = sum cross(r(k), g(k))
// Segments within one body keep their length: their terms cancel.
//
// Friction at inner point k acts on the length of the path up to k (its first k
// segments): rope stuck at k keeps it, rope slipping through k changes it. Its impulse
// pulls those segments tighter (or slacker) than the ones after k, bounded by the
// capstan equation on the tension beyond k: T(before) <= T(after) e^(friction wrap),
// T(before) >= T(after) e^(-friction wrap). Which body holds k makes no difference.

typedef struct b3RopeJacobian
{
	b3Vec3 linear[B3_MAX_ROPE_BODIES];
	b3Vec3 angular[B3_MAX_ROPE_BODIES];
} b3RopeJacobian;

typedef struct b3RopePath
{
	// Points relative to their body's center, and each segment's unit direction (zero
	// when it has no length)
	b3Vec3 r[B3_MAX_ROPE_POINTS];
	b3Vec3 u[B3_MAX_ROPE_POINTS];
	float segment[B3_MAX_ROPE_POINTS];
	float length;
} b3RopePath;

// The bodies' velocities while a rope solves: copies of their states', written back to the
// dynamic ones at the end.
typedef struct b3RopeVelocities
{
	b3Vec3 v[B3_MAX_ROPE_BODIES];
	b3Vec3 w[B3_MAX_ROPE_BODIES];
} b3RopeVelocities;

// The path from the step's start (anchors, deltaCenter) moved by the bodies' deltas.
static b3RopePath b3RopePathOf( const b3RopeStep* step, b3BodyState* const* states )
{
	const b3RopeData* rope = step->rope;
	b3RopePath path = { 0 };
	b3Vec3 p[B3_MAX_ROPE_POINTS];
	b3Vec3 bases[B3_MAX_ROPE_BODIES];
	for ( int k = 0; k < step->bodyCount; ++k )
	{
		bases[k] = b3Add( step->deltaCenter[k], states[k]->deltaPosition );
	}
	int count = rope->pointCount;
	for ( int i = 0; i < count; ++i )
	{
		int k = rope->pointBodies[i];
		path.r[i] = b3RotateVector( states[k]->deltaRotation, step->anchors[i] );
		p[i] = b3Add( bases[k], path.r[i] );
	}
	for ( int i = 0; i + 1 < count; ++i )
	{
		b3Vec3 d = b3Sub( p[i + 1], p[i] );
		float length = b3Length( d );
		path.segment[i] = length;
		path.length += length;
		path.u[i] = length > FLT_EPSILON ? b3MulSV( 1.0f / length, d ) : b3Vec3_zero;
	}
	return path;
}

static void b3RopeAddPoint( const b3RopeData* rope, b3RopeJacobian* j, const b3RopePath* path, int i, b3Vec3 linear )
{
	int k = rope->pointBodies[i];
	j->linear[k] = b3Add( j->linear[k], linear );
	j->angular[k] = b3Add( j->angular[k], b3Cross( path->r[i], linear ) );
}

// Jacobian of the length of the path's first `points` points (the whole rope at
// pointCount).
static b3RopeJacobian b3RopeJacobianOf( const b3RopeData* rope, const b3RopePath* path, int points )
{
	b3RopeJacobian j = { 0 };
	b3Vec3 previous = b3Vec3_zero;
	for ( int i = 0; i < points; ++i )
	{
		b3Vec3 next = i + 1 < points ? path->u[i] : b3Vec3_zero;
		b3RopeAddPoint( rope, &j, path, i, b3Sub( previous, next ) );
		previous = next;
	}
	return j;
}

static float b3RopeInvMass( const b3RopeStep* step, const b3RopeJacobian* j )
{
	float invMass = 0.0f;
	for ( int k = 0; k < step->bodyCount; ++k )
	{
		invMass += step->invMass[k] * b3Dot( j->linear[k], j->linear[k] ) + b3Dot( j->angular[k], b3MulMV( step->invI[k], j->angular[k] ) );
	}
	return invMass;
}

// The rate the length lengthens.
static float b3RopeRate( const b3RopeStep* step, const b3RopeJacobian* j, const b3RopeVelocities* vel )
{
	float rate = 0.0f;
	for ( int k = 0; k < step->bodyCount; ++k )
	{
		rate += b3Dot( j->linear[k], vel->v[k] ) + b3Dot( j->angular[k], vel->w[k] );
	}
	return rate;
}

// Velocities change by -invM J^T impulse: the rope pulls its points toward each other.
static void b3RopeApply( const b3RopeStep* step, const b3RopeJacobian* j, float impulse, b3RopeVelocities* vel )
{
	for ( int k = 0; k < step->bodyCount; ++k )
	{
		vel->v[k] = b3MulSub( vel->v[k], step->invMass[k] * impulse, j->linear[k] );
		vel->w[k] = b3Sub( vel->w[k], b3MulMV( step->invI[k], b3MulSV( impulse, j->angular[k] ) ) );
	}
}

// Bending: segment i turning on the body at its start (or end), about an axis e across
// it. Its own angular velocity about e is dot(e, cross(u, v(i+1) - v(i))) / segment,
// that is dot(cross(e, u), v(i+1) - v(i)) / segment; the body's is dot(e, w).
// Cdot = dot(t, v(i+1) - v(i)) - dot(e, w), t = cross(e, u) / segment
// C = dot(e, cross(rest, u)), the angle turned from the rest direction about e
// Only segments between two bodies turn on them.
static bool b3RopeBends( const b3RopeData* rope, int segment )
{
	return rope->pointBodies[segment] != rope->pointBodies[segment + 1];
}

static b3RopeJacobian b3RopeBendJacobian( const b3RopeData* rope, const b3RopePath* path, int segment, bool atEnd, b3Vec3 e )
{
	b3RopeJacobian j = { 0 };
	b3Vec3 t = b3MulSV( 1.0f / path->segment[segment], b3Cross( e, path->u[segment] ) );
	b3RopeAddPoint( rope, &j, path, segment, b3Neg( t ) );
	b3RopeAddPoint( rope, &j, path, segment + 1, t );
	int k = rope->pointBodies[atEnd ? segment + 1 : segment];
	j.angular[k] = b3Sub( j.angular[k], e );
	return j;
}

// Apply the bending impulses held (warm start), or solve them: a spring-damper of
// bendStiffness / segment and bendDamping / segment per radian, implicit over the substep.
static void b3RopeBend( const b3RopeStep* step, const b3RopePath* path, b3BodyState* const* states, float h, bool solve,
						b3RopeVelocities* vel )
{
	b3RopeData* rope = step->rope;
	for ( int i = 0; i + 1 < rope->pointCount; ++i )
	{
		if ( b3RopeBends( rope, i ) == false || path->segment[i] < B3_LINEAR_SLOP )
		{
			continue;
		}
		b3Vec3 u = path->u[i];
		b3Vec3 e1 = b3Perp( u );
		b3Vec3 axes[2] = { e1, b3Cross( u, e1 ) };
		float k = rope->bendStiffness / path->segment[i];
		float c = rope->bendDamping / path->segment[i];
		float gamma = 1.0f / ( h * ( c + h * k ) );
		float biasRate = k / ( c + h * k );
		for ( int end = 0; end < 2; ++end )
		{
			bool atEnd = end == 1;
			b3Vec3* held = atEnd ? rope->bendEndImpulse + i : rope->bendStartImpulse + i;
			int body = rope->pointBodies[atEnd ? i + 1 : i];
			b3Vec3 rest = b3RotateVector( states[body]->deltaRotation, atEnd ? step->restEndWorld[i] : step->restStartWorld[i] );
			b3Vec3 bent = b3Cross( rest, u );
			b3Vec3 total = b3Vec3_zero;
			for ( int a = 0; a < 2; ++a )
			{
				b3Vec3 e = axes[a];
				b3RopeJacobian j = b3RopeBendJacobian( rope, path, i, atEnd, e );
				float impulse = b3Dot( *held, e );
				if ( solve )
				{
					float invMass = b3RopeInvMass( step, &j );
					float Cdot = b3RopeRate( step, &j, vel );
					float delta = ( Cdot + biasRate * b3Dot( bent, e ) - gamma * impulse ) / ( invMass + gamma );
					b3RopeApply( step, &j, delta, vel );
					impulse += delta;
				}
				else
				{
					b3RopeApply( step, &j, impulse, vel );
				}
				total = b3MulAdd( total, impulse, e );
			}
			*held = total;
		}
	}
}

static bool b3RopeBendActive( const b3RopeData* rope )
{
	return rope->bendStiffness > 0.0f || rope->bendDamping > 0.0f;
}

static void b3RopeBendRelease( b3RopeData* rope )
{
	for ( int i = 0; i < B3_MAX_ROPE_POINTS; ++i )
	{
		rope->bendStartImpulse[i] = b3Vec3_zero;
		rope->bendEndImpulse[i] = b3Vec3_zero;
	}
}

// The states of the rope's bodies (identity in `dummies` for bodies not awake) and their
// velocities.
static void b3RopeStates( const b3RopeStep* step, b3StepContext* context, b3BodyState* dummies, b3BodyState** states,
						  b3RopeVelocities* vel )
{
	for ( int k = 0; k < step->bodyCount; ++k )
	{
		dummies[k] = b3_identityBodyState;
		states[k] = step->index[k] == B3_NULL_INDEX ? dummies + k : context->states + step->index[k];
		vel->v[k] = states[k]->linearVelocity;
		vel->w[k] = states[k]->angularVelocity;
	}
}

static void b3RopeStore( const b3RopeStep* step, b3BodyState** states, const b3RopeVelocities* vel )
{
	for ( int k = 0; k < step->bodyCount; ++k )
	{
		if ( states[k]->flags & b3_dynamicFlag )
		{
			states[k]->linearVelocity = vel->v[k];
			states[k]->angularVelocity = vel->w[k];
		}
	}
}

static b3RopeStep* b3RopeStepOf( b3JointSim* base, b3StepContext* context )
{
	B3_ASSERT( 0 <= base->ropeJoint.ropeId && base->ropeJoint.ropeId < context->ropeStepCount );
	return context->ropeSteps + base->ropeJoint.ropeId;
}

void b3PrepareRopeJoint( b3JointSim* base, b3StepContext* context )
{
	B3_ASSERT( base->type == b3_ropeJoint );

	b3World* world = context->world;
	b3RopeData* rope = b3GetRopeData( world, base );

	// One step entry per rope id: ropes prepared in parallel colors write their own
	b3RopeStep* step = b3RopeStepOf( base, context );

	step->rope = rope;
	step->bodyCount = rope->bodyCount;

	b3BodySim* sims[B3_MAX_ROPE_BODIES];
	bool anyAwake = false;
	for ( int k = 0; k < rope->bodyCount; ++k )
	{
		b3Body* body = b3Array_Get( world->bodies, rope->bodyIds[k] );
		b3SolverSet* set = b3Array_Get( world->solverSets, body->setIndex );
		sims[k] = b3Array_Get( set->bodySims, body->localIndex );
		bool awake = body->setIndex == b3_awakeSet;
		anyAwake = anyAwake || awake;
		step->index[k] = awake ? body->localIndex : B3_NULL_INDEX;

		// A disabled body takes no part: it holds its points where it is
		bool disabled = body->setIndex == b3_disabledSet;
		step->invMass[k] = disabled ? 0.0f : sims[k]->invMass;
		step->invI[k] = disabled ? b3Mat3_zero : sims[k]->invInertiaWorld;
		step->deltaCenter[k] = b3SubPos( sims[k]->center, sims[0]->center );
	}
	B3_ASSERT( anyAwake );
	B3_UNUSED( anyAwake );

	// The base joint's own view, for generic queries
	base->invMassA = step->invMass[0];
	base->invMassB = step->invMass[1];
	base->invIA = step->invI[0];
	base->invIB = step->invI[1];

	for ( int i = 0; i < rope->pointCount; ++i )
	{
		b3BodySim* sim = sims[rope->pointBodies[i]];
		step->anchors[i] = b3RotateVector( sim->transform.q, b3Sub( rope->localPoints[i], sim->localCenter ) );
	}

	b3BodyState identity[B3_MAX_ROPE_BODIES];
	b3BodyState* states[B3_MAX_ROPE_BODIES];
	for ( int k = 0; k < rope->bodyCount; ++k )
	{
		identity[k] = b3_identityBodyState;
		states[k] = identity + k;
	}
	b3RopePath path = b3RopePathOf( step, states );

	// Taut, the rope is a spring of rate EA / L on the path's coordinate: its natural
	// frequency follows from the effective mass there. Clamped to the step like other
	// joint springs; a stiffer rope than the step resolves stays as stiff as it can.
	step->soft = false;
	if ( rope->stiffness > 0.0f )
	{
		b3RopeJacobian j = b3RopeJacobianOf( rope, &path, rope->pointCount );
		float invMass = b3RopeInvMass( step, &j );
		if ( invMass > 0.0f )
		{
			float rate = rope->stiffness / rope->length;
			float hertz = sqrtf( rate * invMass ) / ( 2.0f * B3_PI );
			hertz = b3MinFloat( hertz, 0.25f * context->inv_h );
			step->softness = b3MakeSoft( hertz, rope->dampingRatio, context->h );
			step->soft = true;
		}
	}

	// The capstan bounds from the angle the path turns through at each inner point. A point
	// on top of its neighbor turns the path through no angle (its segment has no direction).
	for ( int k = 1; k + 1 < rope->pointCount; ++k )
	{
		step->gripTighter[k] = 0.0f;
		step->gripSlacker[k] = 0.0f;
		rope->slipEnergy[k] = 0.0f;
		float wrap = 0.0f;
		if ( path.segment[k - 1] >= B3_LINEAR_SLOP && path.segment[k] >= B3_LINEAR_SLOP )
		{
			wrap = acosf( b3ClampFloat( b3Dot( path.u[k - 1], path.u[k] ), -1.0f, 1.0f ) );
		}
		float grip = rope->friction[k] * wrap;
		if ( grip > 0.0f )
		{
			step->gripTighter[k] = expf( grip ) - 1.0f;
			step->gripSlacker[k] = 1.0f - expf( -grip );
		}
		else
		{
			rope->frictionImpulse[k] = 0.0f;
		}
	}

	// The rest directions, carried by their bodies.
	for ( int i = 0; i + 1 < rope->pointCount; ++i )
	{
		b3Quat qStart = sims[rope->pointBodies[i]]->transform.q;
		b3Quat qEnd = sims[rope->pointBodies[i + 1]]->transform.q;
		step->restStartWorld[i] = b3RotateVector( qStart, rope->restStart[i] );
		step->restEndWorld[i] = b3RotateVector( qEnd, rope->restEnd[i] );
	}

	if ( context->enableWarmStarting == false )
	{
		rope->impulse = 0.0f;
		for ( int k = 0; k < B3_MAX_ROPE_POINTS; ++k )
		{
			rope->frictionImpulse[k] = 0.0f;
		}
	}
	if ( context->enableWarmStarting == false || b3RopeBendActive( rope ) == false )
	{
		b3RopeBendRelease( rope );
	}
}

void b3WarmStartRopeJoint( b3JointSim* base, b3StepContext* context )
{
	B3_ASSERT( base->type == b3_ropeJoint );

	b3RopeStep* step = b3RopeStepOf( base, context );
	b3RopeData* rope = step->rope;

	b3BodyState dummies[B3_MAX_ROPE_BODIES];
	b3BodyState* states[B3_MAX_ROPE_BODIES];
	b3RopeVelocities vel;
	b3RopeStates( step, context, dummies, states, &vel );

	b3RopePath path = b3RopePathOf( step, states );

	b3RopeJacobian j = b3RopeJacobianOf( rope, &path, rope->pointCount );
	b3RopeApply( step, &j, rope->impulse, &vel );
	for ( int k = 1; k + 1 < rope->pointCount; ++k )
	{
		if ( rope->frictionImpulse[k] != 0.0f )
		{
			b3RopeJacobian jk = b3RopeJacobianOf( rope, &path, k + 1 );
			b3RopeApply( step, &jk, rope->frictionImpulse[k], &vel );
		}
	}
	if ( b3RopeBendActive( rope ) )
	{
		b3RopeBend( step, &path, states, context->h, false, &vel );
	}

	b3RopeStore( step, states, &vel );
}

void b3SolveRopeJoint( b3JointSim* base, b3StepContext* context, bool useBias )
{
	B3_ASSERT( base->type == b3_ropeJoint );

	b3RopeStep* step = b3RopeStepOf( base, context );
	b3RopeData* rope = step->rope;

	b3BodyState dummies[B3_MAX_ROPE_BODIES];
	b3BodyState* states[B3_MAX_ROPE_BODIES];
	b3RopeVelocities vel;
	b3RopeStates( step, context, dummies, states, &vel );

	b3RopePath path = b3RopePathOf( step, states );

	// Friction first, bounded by the tension beyond each point (like contact friction by
	// the normal impulse); from the far end in, so each bound sees this pass's impulses.
	for ( int k = rope->pointCount - 2; k >= 1; --k )
	{
		if ( step->gripTighter[k] == 0.0f )
		{
			continue;
		}
		b3RopeJacobian jk = b3RopeJacobianOf( rope, &path, k + 1 );
		float invMass = b3RopeInvMass( step, &jk );
		if ( invMass == 0.0f )
		{
			continue;
		}
		float beyond = b3RopeSegmentImpulse( rope, k );
		float impulse = b3RopeRate( step, &jk, &vel ) / invMass;
		float newImpulse =
			b3ClampFloat( rope->frictionImpulse[k] + impulse, -step->gripSlacker[k] * beyond, step->gripTighter[k] * beyond );
		impulse = newImpulse - rope->frictionImpulse[k];
		rope->frictionImpulse[k] = newImpulse;
		b3RopeApply( step, &jk, impulse, &vel );
	}

	b3RopeJacobian j = b3RopeJacobianOf( rope, &path, rope->pointCount );
	float invMass = b3RopeInvMass( step, &j );
	if ( invMass > 0.0f )
	{
		// Slack: speculative. Taut: the rope's own spring, or (inextensible) the joint
		// softness while biased, like a limit.
		float slack = rope->length - path.length;
		float bias = 0.0f;
		float massScale = 1.0f;
		float impulseScale = 0.0f;
		if ( slack > 0.0f )
		{
			bias = slack * context->inv_h;
		}
		else if ( step->soft )
		{
			bias = step->softness.biasRate * slack;
			massScale = step->softness.massScale;
			impulseScale = step->softness.impulseScale;
		}
		else if ( useBias )
		{
			bias = base->constraintSoftness.biasRate * slack;
			massScale = base->constraintSoftness.massScale;
			impulseScale = base->constraintSoftness.impulseScale;
		}

		// Rate the slack opens: -Cdot
		float Cdot = -b3RopeRate( step, &j, &vel );
		float impulse = -massScale * ( Cdot + bias ) / invMass - impulseScale * rope->impulse;
		float newImpulse = b3MaxFloat( 0.0f, rope->impulse + impulse );
		impulse = newImpulse - rope->impulse;
		rope->impulse = newImpulse;

		b3RopeApply( step, &j, impulse, &vel );
	}

	// Bending, while taut: slack rope hangs as it likes.
	if ( b3RopeBendActive( rope ) )
	{
		if ( path.length < rope->length - B3_LINEAR_SLOP )
		{
			b3RopeBendRelease( rope );
		}
		else
		{
			b3RopeBend( step, &path, states, context->h, true, &vel );
		}
	}

	// Relaxed (each substep's final velocities): the work friction does against the slip,
	// summed over the step.
	if ( useBias == false )
	{
		for ( int k = 1; k + 1 < rope->pointCount; ++k )
		{
			if ( step->gripTighter[k] == 0.0f )
			{
				continue;
			}
			b3RopeJacobian jk = b3RopeJacobianOf( rope, &path, k + 1 );
			rope->slipEnergy[k] += b3AbsFloat( rope->frictionImpulse[k] * b3RopeRate( step, &jk, &vel ) );
		}
	}

	b3RopeStore( step, states, &vel );
}

void b3DrawRopeJoint( b3DebugDraw* draw, b3World* world, b3JointSim* base )
{
	B3_ASSERT( base->type == b3_ropeJoint );

	b3RopeData* rope = b3GetRopeData( world, base );
	b3Pos points[B3_MAX_ROPE_POINTS];
	int count = b3RopeWorldPoints( world, rope, points, B3_MAX_ROPE_POINTS );
	for ( int i = 0; i < count; ++i )
	{
		if ( i > 0 )
		{
			draw->DrawSegmentFcn( points[i - 1], points[i], rope->impulse > 0.0f ? b3_colorOrange : b3_colorWhite, draw->context );
		}
		draw->DrawPointFcn( points[i], 4.0f, b3_colorWhite, draw->context );
	}
}
