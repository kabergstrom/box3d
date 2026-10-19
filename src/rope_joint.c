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

void b3RopeJoint_SetLength( b3JointId jointId, float length )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetLength, jointId, length );
	b3JointSim* base = b3GetJointSimCheckType( jointId, b3_ropeJoint );
	base->ropeJoint.length = b3ClampFloat( length, B3_LINEAR_SLOP, B3_HUGE );
}

float b3RopeJoint_GetLength( b3JointId jointId )
{
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	return base->ropeJoint.length;
}

void b3RopeJoint_SetStiffness( b3JointId jointId, float stiffness )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetStiffness, jointId, stiffness );
	b3JointSim* base = b3GetJointSimCheckType( jointId, b3_ropeJoint );
	base->ropeJoint.stiffness = b3MaxFloat( stiffness, 0.0f );
}

float b3RopeJoint_GetStiffness( b3JointId jointId )
{
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	return base->ropeJoint.stiffness;
}

void b3RopeJoint_SetDampingRatio( b3JointId jointId, float dampingRatio )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetDampingRatio, jointId, dampingRatio );
	b3JointSim* base = b3GetJointSimCheckType( jointId, b3_ropeJoint );
	base->ropeJoint.dampingRatio = b3MaxFloat( dampingRatio, 0.0f );
}

float b3RopeJoint_GetDampingRatio( b3JointId jointId )
{
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	return base->ropeJoint.dampingRatio;
}

void b3RopeJoint_SetBendStiffness( b3JointId jointId, float bendStiffness )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetBendStiffness, jointId, bendStiffness );
	b3JointSim* base = b3GetJointSimCheckType( jointId, b3_ropeJoint );
	base->ropeJoint.bendStiffness = b3MaxFloat( bendStiffness, 0.0f );
}

float b3RopeJoint_GetBendStiffness( b3JointId jointId )
{
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	return base->ropeJoint.bendStiffness;
}

void b3RopeJoint_SetBendDamping( b3JointId jointId, float bendDamping )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetBendDamping, jointId, bendDamping );
	b3JointSim* base = b3GetJointSimCheckType( jointId, b3_ropeJoint );
	base->ropeJoint.bendDamping = b3MaxFloat( bendDamping, 0.0f );
}

float b3RopeJoint_GetBendDamping( b3JointId jointId )
{
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	return base->ropeJoint.bendDamping;
}

void b3RopeJoint_SetRestDirections( b3JointId jointId, int segment, b3Vec3 start, b3Vec3 end )
{
	b3World* world = b3GetWorld( jointId.world0 );
	B3_REC( world, RopeJointSetRestDirections, jointId, segment, start, end );
	b3JointSim* base = b3GetJointSimCheckType( jointId, b3_ropeJoint );
	b3RopeJoint* joint = &base->ropeJoint;
	if ( segment < 0 || segment + 1 >= joint->pointCount || b3IsValidVec3( start ) == false || b3IsValidVec3( end ) == false )
	{
		return;
	}
	joint->restStart[segment] = start;
	joint->restEnd[segment] = end;
}

int b3RopeJoint_GetRestDirections( b3JointId jointId, b3Vec3* start, b3Vec3* end, int capacity )
{
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	if ( start == NULL || end == NULL )
	{
		return 0;
	}
	int count = 0;
	for ( ; count + 1 < base->ropeJoint.pointCount && count < capacity; ++count )
	{
		start[count] = base->ropeJoint.restStart[count];
		end[count] = base->ropeJoint.restEnd[count];
	}
	return count;
}

// World points of the path, at most capacity.
static int b3RopeWorldPoints( b3World* world, b3JointSim* base, b3Pos* points, int capacity )
{
	b3RopeJoint* joint = &base->ropeJoint;
	b3WorldTransform transformA = b3GetBodyTransform( world, base->bodyIdA );
	b3WorldTransform transformB = b3GetBodyTransform( world, base->bodyIdB );
	int count = joint->pointCount < capacity ? joint->pointCount : capacity;
	for ( int i = 0; i < count; ++i )
	{
		b3WorldTransform xf = ( joint->bodyBMask >> i ) & 1u ? transformB : transformA;
		points[i] = b3TransformWorldPoint( xf, joint->localPoints[i] );
	}
	return count;
}

float b3GetRopeJointLength( b3World* world, b3JointSim* base )
{
	b3Pos points[B3_MAX_ROPE_POINTS];
	int count = b3RopeWorldPoints( world, base, points, B3_MAX_ROPE_POINTS );
	float length = 0.0f;
	for ( int i = 0; i + 1 < count; ++i )
	{
		length += b3Length( b3SubPos( points[i + 1], points[i] ) );
	}
	return length;
}

void b3SetRopeJointRest( b3World* world, b3JointSim* base )
{
	b3RopeJoint* joint = &base->ropeJoint;
	b3Pos points[B3_MAX_ROPE_POINTS];
	int count = b3RopeWorldPoints( world, base, points, B3_MAX_ROPE_POINTS );
	b3Quat qA = b3GetBodyTransform( world, base->bodyIdA ).q;
	b3Quat qB = b3GetBodyTransform( world, base->bodyIdB ).q;
	for ( int i = 0; i + 1 < count; ++i )
	{
		b3Vec3 d = b3SubPos( points[i + 1], points[i] );
		float length = b3Length( d );
		b3Vec3 u = length > FLT_EPSILON ? b3MulSV( 1.0f / length, d ) : b3Vec3_zero;
		joint->restStart[i] = b3InvRotateVector( ( joint->bodyBMask >> i ) & 1u ? qB : qA, u );
		joint->restEnd[i] = b3InvRotateVector( ( joint->bodyBMask >> ( i + 1 ) ) & 1u ? qB : qA, u );
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
static float b3RopeSegmentImpulse( const b3RopeJoint* joint, int segment )
{
	float impulse = joint->impulse;
	for ( int k = segment + 1; k + 1 < joint->pointCount; ++k )
	{
		impulse += joint->frictionImpulse[k];
	}
	return impulse;
}

float b3GetRopeJointMaxImpulse( const b3RopeJoint* joint )
{
	float most = 0.0f;
	for ( int i = 0; i + 1 < joint->pointCount; ++i )
	{
		most = b3MaxFloat( most, b3RopeSegmentImpulse( joint, i ) );
	}
	return most;
}

float b3RopeJoint_GetTension( b3JointId jointId )
{
	b3World* world = b3GetWorld( jointId.world0 );
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	return world->inv_h * b3GetRopeJointMaxImpulse( &base->ropeJoint );
}

int b3RopeJoint_GetSegmentTensions( b3JointId jointId, float* tensions, int capacity )
{
	b3World* world = b3GetWorld( jointId.world0 );
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	if ( tensions == NULL )
	{
		return 0;
	}
	int count = 0;
	for ( ; count + 1 < base->ropeJoint.pointCount && count < capacity; ++count )
	{
		tensions[count] = world->inv_h * b3RopeSegmentImpulse( &base->ropeJoint, count );
	}
	return count;
}

int b3RopeJoint_GetSlipEnergies( b3JointId jointId, float* energies, int capacity )
{
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	if ( energies == NULL )
	{
		return 0;
	}
	int count = base->ropeJoint.pointCount < capacity ? base->ropeJoint.pointCount : capacity;
	for ( int i = 0; i < count; ++i )
	{
		energies[i] = base->ropeJoint.slipEnergy[i];
	}
	return count;
}

int b3RopeJoint_GetPointCount( b3JointId jointId )
{
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	return base->ropeJoint.pointCount;
}

int b3RopeJoint_GetWorldPoints( b3JointId jointId, b3Pos* points, int capacity )
{
	b3JointSim* base = b3ReadJointSimCheckType( jointId, b3_ropeJoint );
	b3World* world = b3GetUnlockedWorld( jointId.world0 );
	if ( world == NULL || points == NULL || capacity <= 0 )
	{
		return 0;
	}
	return b3RopeWorldPoints( world, base, points, capacity );
}

// The rope's pull on body B: every segment pulls its ends toward each other.
b3Vec3 b3GetRopeJointForce( b3World* world, b3JointSim* base )
{
	b3RopeJoint* joint = &base->ropeJoint;
	b3Pos points[B3_MAX_ROPE_POINTS];
	int count = b3RopeWorldPoints( world, base, points, B3_MAX_ROPE_POINTS );
	b3Vec3 force = b3Vec3_zero;
	for ( int i = 0; i < count; ++i )
	{
		if ( ( ( joint->bodyBMask >> i ) & 1u ) == 0 )
		{
			continue;
		}
		if ( i > 0 )
		{
			float tension = world->inv_h * b3RopeSegmentImpulse( joint, i - 1 );
			force = b3MulAdd( force, tension, b3Normalize( b3SubPos( points[i - 1], points[i] ) ) );
		}
		if ( i + 1 < count )
		{
			float tension = world->inv_h * b3RopeSegmentImpulse( joint, i );
			force = b3MulAdd( force, tension, b3Normalize( b3SubPos( points[i + 1], points[i] ) ) );
		}
	}
	return force;
}

// C = sum |p(i+1) - p(i)| - L <= 0
// u(i) = (p(i+1) - p(i)) / |p(i+1) - p(i)|
// Cdot = sum over points k of dot(g(k), v(k)), g(k) = u(k-1) - u(k) (missing terms zero)
// v(k) = v + cross(w, r(k)) of the point's body
// J = [JvA JwA JvB JwB], JvX = sum g(k) over X's points, JwX = sum cross(r(k), g(k))
// Segments within one body keep their length: their terms cancel.
//
// Friction at inner point k acts on the length of the path up to k (its first k
// segments): rope stuck at k keeps it, rope slipping through k changes it. Its impulse
// pulls those segments tighter (or slacker) than the ones after k, bounded by the
// capstan equation on the tension beyond k: T(before) <= T(after) e^(friction wrap),
// T(before) >= T(after) e^(-friction wrap).

typedef struct b3RopeJacobian
{
	b3Vec3 linearA, angularA;
	b3Vec3 linearB, angularB;
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

// The path from the step's start (anchors, deltaCenter) moved by the bodies' deltas.
static b3RopePath b3RopePathOf( const b3RopeJoint* joint, const b3BodyState* stateA, const b3BodyState* stateB )
{
	b3RopePath path = { 0 };
	b3Vec3 p[B3_MAX_ROPE_POINTS];
	b3Vec3 baseB = b3Add( joint->deltaCenter, stateB->deltaPosition );
	int count = joint->pointCount;
	for ( int i = 0; i < count; ++i )
	{
		if ( ( joint->bodyBMask >> i ) & 1u )
		{
			path.r[i] = b3RotateVector( stateB->deltaRotation, joint->anchors[i] );
			p[i] = b3Add( baseB, path.r[i] );
		}
		else
		{
			path.r[i] = b3RotateVector( stateA->deltaRotation, joint->anchors[i] );
			p[i] = b3Add( stateA->deltaPosition, path.r[i] );
		}
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

// Jacobian of the length of the path's first `points` points (the whole rope at
// pointCount).
static b3RopeJacobian b3RopeJacobianOf( const b3RopeJoint* joint, const b3RopePath* path, int points )
{
	b3RopeJacobian j = { 0 };
	b3Vec3 previous = b3Vec3_zero;
	for ( int i = 0; i < points; ++i )
	{
		b3Vec3 next = i + 1 < points ? path->u[i] : b3Vec3_zero;
		b3Vec3 g = b3Sub( previous, next );
		if ( ( joint->bodyBMask >> i ) & 1u )
		{
			j.linearB = b3Add( j.linearB, g );
			j.angularB = b3Add( j.angularB, b3Cross( path->r[i], g ) );
		}
		else
		{
			j.linearA = b3Add( j.linearA, g );
			j.angularA = b3Add( j.angularA, b3Cross( path->r[i], g ) );
		}
		previous = next;
	}
	return j;
}

static float b3RopeInvMass( const b3JointSim* base, const b3RopeJacobian* j )
{
	return base->invMassA * b3Dot( j->linearA, j->linearA ) + b3Dot( j->angularA, b3MulMV( base->invIA, j->angularA ) ) +
		   base->invMassB * b3Dot( j->linearB, j->linearB ) + b3Dot( j->angularB, b3MulMV( base->invIB, j->angularB ) );
}

// The rate the length lengthens.
static float b3RopeRate( const b3RopeJacobian* j, b3Vec3 vA, b3Vec3 wA, b3Vec3 vB, b3Vec3 wB )
{
	return b3Dot( j->linearA, vA ) + b3Dot( j->angularA, wA ) + b3Dot( j->linearB, vB ) + b3Dot( j->angularB, wB );
}

// Velocities change by -invM J^T impulse: the rope pulls its points toward each other.
static void b3RopeApply( const b3JointSim* base, const b3RopeJacobian* j, float impulse, b3Vec3* vA, b3Vec3* wA, b3Vec3* vB,
						 b3Vec3* wB )
{
	*vA = b3MulSub( *vA, base->invMassA * impulse, j->linearA );
	*wA = b3Sub( *wA, b3MulMV( base->invIA, b3MulSV( impulse, j->angularA ) ) );
	*vB = b3MulSub( *vB, base->invMassB * impulse, j->linearB );
	*wB = b3Sub( *wB, b3MulMV( base->invIB, b3MulSV( impulse, j->angularB ) ) );
}

// Bending: segment i turning on the body at its start (or end), about an axis e across
// it. Its own angular velocity about e is dot(e, cross(u, v(i+1) - v(i))) / segment,
// that is dot(cross(e, u), v(i+1) - v(i)) / segment; the body's is dot(e, w).
// Cdot = dot(t, v(i+1) - v(i)) - dot(e, w), t = cross(e, u) / segment
// C = dot(e, cross(rest, u)), the angle turned from the rest direction about e
// Only segments between the two bodies turn on them.
static bool b3RopeBends( const b3RopeJoint* joint, int segment )
{
	return ( ( joint->bodyBMask >> segment ) & 1u ) != ( ( joint->bodyBMask >> ( segment + 1 ) ) & 1u );
}

static void b3RopeAddPoint( const b3RopeJoint* joint, b3RopeJacobian* j, const b3RopePath* path, int i, b3Vec3 linear )
{
	if ( ( joint->bodyBMask >> i ) & 1u )
	{
		j->linearB = b3Add( j->linearB, linear );
		j->angularB = b3Add( j->angularB, b3Cross( path->r[i], linear ) );
	}
	else
	{
		j->linearA = b3Add( j->linearA, linear );
		j->angularA = b3Add( j->angularA, b3Cross( path->r[i], linear ) );
	}
}

static b3RopeJacobian b3RopeBendJacobian( const b3RopeJoint* joint, const b3RopePath* path, int segment, bool atEnd, b3Vec3 e )
{
	b3RopeJacobian j = { 0 };
	b3Vec3 t = b3MulSV( 1.0f / path->segment[segment], b3Cross( e, path->u[segment] ) );
	b3RopeAddPoint( joint, &j, path, segment, b3Neg( t ) );
	b3RopeAddPoint( joint, &j, path, segment + 1, t );
	if ( ( joint->bodyBMask >> ( atEnd ? segment + 1 : segment ) ) & 1u )
	{
		j.angularB = b3Sub( j.angularB, e );
	}
	else
	{
		j.angularA = b3Sub( j.angularA, e );
	}
	return j;
}

// Apply the bending impulses held (warm start), or solve them: a spring-damper of
// bendStiffness / segment and bendDamping / segment per radian, implicit over the substep.
static void b3RopeBend( const b3JointSim* base, b3RopeJoint* joint, const b3RopePath* path, const b3BodyState* stateA,
						const b3BodyState* stateB, float h, bool solve, b3Vec3* vA, b3Vec3* wA, b3Vec3* vB, b3Vec3* wB )
{
	for ( int i = 0; i + 1 < joint->pointCount; ++i )
	{
		if ( b3RopeBends( joint, i ) == false || path->segment[i] < B3_LINEAR_SLOP )
		{
			continue;
		}
		b3Vec3 u = path->u[i];
		b3Vec3 e1 = b3Perp( u );
		b3Vec3 axes[2] = { e1, b3Cross( u, e1 ) };
		float k = joint->bendStiffness / path->segment[i];
		float c = joint->bendDamping / path->segment[i];
		float gamma = 1.0f / ( h * ( c + h * k ) );
		float biasRate = k / ( c + h * k );
		for ( int end = 0; end < 2; ++end )
		{
			bool atEnd = end == 1;
			b3Vec3* held = atEnd ? joint->bendEndImpulse + i : joint->bendStartImpulse + i;
			bool onB = ( joint->bodyBMask >> ( atEnd ? i + 1 : i ) ) & 1u;
			b3Vec3 rest = b3RotateVector( onB ? stateB->deltaRotation : stateA->deltaRotation,
										  atEnd ? joint->restEndWorld[i] : joint->restStartWorld[i] );
			b3Vec3 bent = b3Cross( rest, u );
			b3Vec3 total = b3Vec3_zero;
			for ( int a = 0; a < 2; ++a )
			{
				b3Vec3 e = axes[a];
				b3RopeJacobian j = b3RopeBendJacobian( joint, path, i, atEnd, e );
				float impulse = b3Dot( *held, e );
				if ( solve )
				{
					float invMass = b3RopeInvMass( base, &j );
					float Cdot = b3RopeRate( &j, *vA, *wA, *vB, *wB );
					float step = ( Cdot + biasRate * b3Dot( bent, e ) - gamma * impulse ) / ( invMass + gamma );
					b3RopeApply( base, &j, step, vA, wA, vB, wB );
					impulse += step;
				}
				else
				{
					b3RopeApply( base, &j, impulse, vA, wA, vB, wB );
				}
				total = b3MulAdd( total, impulse, e );
			}
			*held = total;
		}
	}
}

static bool b3RopeBendActive( const b3RopeJoint* joint )
{
	return joint->bendStiffness > 0.0f || joint->bendDamping > 0.0f;
}

static void b3RopeBendRelease( b3RopeJoint* joint )
{
	for ( int i = 0; i < B3_MAX_ROPE_POINTS; ++i )
	{
		joint->bendStartImpulse[i] = b3Vec3_zero;
		joint->bendEndImpulse[i] = b3Vec3_zero;
	}
}

void b3PrepareRopeJoint( b3JointSim* base, b3StepContext* context )
{
	B3_ASSERT( base->type == b3_ropeJoint );

	b3World* world = context->world;
	b3Body* bodyA = b3Array_Get( world->bodies, base->bodyIdA );
	b3Body* bodyB = b3Array_Get( world->bodies, base->bodyIdB );

	B3_ASSERT( bodyA->setIndex == b3_awakeSet || bodyB->setIndex == b3_awakeSet );

	b3SolverSet* setA = b3Array_Get( world->solverSets, bodyA->setIndex );
	b3SolverSet* setB = b3Array_Get( world->solverSets, bodyB->setIndex );

	int localIndexA = bodyA->localIndex;
	int localIndexB = bodyB->localIndex;

	b3BodySim* bodySimA = b3Array_Get( setA->bodySims, localIndexA );
	b3BodySim* bodySimB = b3Array_Get( setB->bodySims, localIndexB );

	base->invMassA = bodySimA->invMass;
	base->invMassB = bodySimB->invMass;
	base->invIA = bodySimA->invInertiaWorld;
	base->invIB = bodySimB->invInertiaWorld;

	b3RopeJoint* joint = &base->ropeJoint;

	joint->indexA = bodyA->setIndex == b3_awakeSet ? localIndexA : B3_NULL_INDEX;
	joint->indexB = bodyB->setIndex == b3_awakeSet ? localIndexB : B3_NULL_INDEX;

	for ( int i = 0; i < joint->pointCount; ++i )
	{
		b3BodySim* sim = ( joint->bodyBMask >> i ) & 1u ? bodySimB : bodySimA;
		joint->anchors[i] = b3RotateVector( sim->transform.q, b3Sub( joint->localPoints[i], sim->localCenter ) );
	}
	joint->deltaCenter = b3SubPos( bodySimB->center, bodySimA->center );

	b3BodyState identity = b3_identityBodyState;
	b3RopePath path = b3RopePathOf( joint, &identity, &identity );

	// Taut, the rope is a spring of rate EA / L on the path's coordinate: its natural
	// frequency follows from the effective mass there. Clamped to the step like other
	// joint springs; a stiffer rope than the step resolves stays as stiff as it can.
	joint->soft = false;
	if ( joint->stiffness > 0.0f )
	{
		b3RopeJacobian j = b3RopeJacobianOf( joint, &path, joint->pointCount );
		float invMass = b3RopeInvMass( base, &j );
		if ( invMass > 0.0f )
		{
			float rate = joint->stiffness / joint->length;
			float hertz = sqrtf( rate * invMass ) / ( 2.0f * B3_PI );
			hertz = b3MinFloat( hertz, 0.25f * context->inv_h );
			joint->softness = b3MakeSoft( hertz, joint->dampingRatio, context->h );
			joint->soft = true;
		}
	}

	// The capstan bounds from the angle the path turns through at each inner point.
	for ( int k = 1; k + 1 < joint->pointCount; ++k )
	{
		joint->gripTighter[k] = 0.0f;
		joint->gripSlacker[k] = 0.0f;
		joint->slipEnergy[k] = 0.0f;
		float wrap = acosf( b3ClampFloat( b3Dot( path.u[k - 1], path.u[k] ), -1.0f, 1.0f ) );
		float grip = joint->friction[k] * wrap;
		if ( grip > 0.0f )
		{
			joint->gripTighter[k] = expf( grip ) - 1.0f;
			joint->gripSlacker[k] = 1.0f - expf( -grip );
		}
		else
		{
			joint->frictionImpulse[k] = 0.0f;
		}
	}

	// The rest directions, carried by their bodies.
	for ( int i = 0; i + 1 < joint->pointCount; ++i )
	{
		b3Quat qStart = ( ( joint->bodyBMask >> i ) & 1u ? bodySimB : bodySimA )->transform.q;
		b3Quat qEnd = ( ( joint->bodyBMask >> ( i + 1 ) ) & 1u ? bodySimB : bodySimA )->transform.q;
		joint->restStartWorld[i] = b3RotateVector( qStart, joint->restStart[i] );
		joint->restEndWorld[i] = b3RotateVector( qEnd, joint->restEnd[i] );
	}

	if ( context->enableWarmStarting == false )
	{
		joint->impulse = 0.0f;
		for ( int k = 0; k < B3_MAX_ROPE_POINTS; ++k )
		{
			joint->frictionImpulse[k] = 0.0f;
		}
	}
	if ( context->enableWarmStarting == false || b3RopeBendActive( joint ) == false )
	{
		b3RopeBendRelease( joint );
	}
}

void b3WarmStartRopeJoint( b3JointSim* base, b3StepContext* context )
{
	B3_ASSERT( base->type == b3_ropeJoint );

	b3BodyState dummyState = b3_identityBodyState;

	b3RopeJoint* joint = &base->ropeJoint;
	b3BodyState* stateA = joint->indexA == B3_NULL_INDEX ? &dummyState : context->states + joint->indexA;
	b3BodyState* stateB = joint->indexB == B3_NULL_INDEX ? &dummyState : context->states + joint->indexB;

	b3RopePath path = b3RopePathOf( joint, stateA, stateB );

	b3Vec3 vA = stateA->linearVelocity;
	b3Vec3 wA = stateA->angularVelocity;
	b3Vec3 vB = stateB->linearVelocity;
	b3Vec3 wB = stateB->angularVelocity;

	b3RopeJacobian j = b3RopeJacobianOf( joint, &path, joint->pointCount );
	b3RopeApply( base, &j, joint->impulse, &vA, &wA, &vB, &wB );
	for ( int k = 1; k + 1 < joint->pointCount; ++k )
	{
		if ( joint->frictionImpulse[k] != 0.0f )
		{
			b3RopeJacobian jk = b3RopeJacobianOf( joint, &path, k + 1 );
			b3RopeApply( base, &jk, joint->frictionImpulse[k], &vA, &wA, &vB, &wB );
		}
	}
	if ( b3RopeBendActive( joint ) )
	{
		b3RopeBend( base, joint, &path, stateA, stateB, context->h, false, &vA, &wA, &vB, &wB );
	}

	if ( stateA->flags & b3_dynamicFlag )
	{
		stateA->linearVelocity = vA;
		stateA->angularVelocity = wA;
	}

	if ( stateB->flags & b3_dynamicFlag )
	{
		stateB->linearVelocity = vB;
		stateB->angularVelocity = wB;
	}
}

void b3SolveRopeJoint( b3JointSim* base, b3StepContext* context, bool useBias )
{
	B3_ASSERT( base->type == b3_ropeJoint );

	b3BodyState dummyState = b3_identityBodyState;

	b3RopeJoint* joint = &base->ropeJoint;
	b3BodyState* stateA = joint->indexA == B3_NULL_INDEX ? &dummyState : context->states + joint->indexA;
	b3BodyState* stateB = joint->indexB == B3_NULL_INDEX ? &dummyState : context->states + joint->indexB;

	b3Vec3 vA = stateA->linearVelocity;
	b3Vec3 wA = stateA->angularVelocity;
	b3Vec3 vB = stateB->linearVelocity;
	b3Vec3 wB = stateB->angularVelocity;

	b3RopePath path = b3RopePathOf( joint, stateA, stateB );

	// Friction first, bounded by the tension beyond each point (like contact friction by
	// the normal impulse); from the far end in, so each bound sees this pass's impulses.
	for ( int k = joint->pointCount - 2; k >= 1; --k )
	{
		if ( joint->gripTighter[k] == 0.0f )
		{
			continue;
		}
		b3RopeJacobian jk = b3RopeJacobianOf( joint, &path, k + 1 );
		float invMass = b3RopeInvMass( base, &jk );
		if ( invMass == 0.0f )
		{
			continue;
		}
		float beyond = b3RopeSegmentImpulse( joint, k );
		float impulse = b3RopeRate( &jk, vA, wA, vB, wB ) / invMass;
		float newImpulse =
			b3ClampFloat( joint->frictionImpulse[k] + impulse, -joint->gripSlacker[k] * beyond, joint->gripTighter[k] * beyond );
		impulse = newImpulse - joint->frictionImpulse[k];
		joint->frictionImpulse[k] = newImpulse;
		b3RopeApply( base, &jk, impulse, &vA, &wA, &vB, &wB );
	}

	b3RopeJacobian j = b3RopeJacobianOf( joint, &path, joint->pointCount );
	float invMass = b3RopeInvMass( base, &j );
	if ( invMass > 0.0f )
	{
		// Slack: speculative. Taut: the rope's own spring, or (inextensible) the joint
		// softness while biased, like a limit.
		float slack = joint->length - path.length;
		float bias = 0.0f;
		float massScale = 1.0f;
		float impulseScale = 0.0f;
		if ( slack > 0.0f )
		{
			bias = slack * context->inv_h;
		}
		else if ( joint->soft )
		{
			bias = joint->softness.biasRate * slack;
			massScale = joint->softness.massScale;
			impulseScale = joint->softness.impulseScale;
		}
		else if ( useBias )
		{
			bias = base->constraintSoftness.biasRate * slack;
			massScale = base->constraintSoftness.massScale;
			impulseScale = base->constraintSoftness.impulseScale;
		}

		// Rate the slack opens: -Cdot
		float Cdot = -b3RopeRate( &j, vA, wA, vB, wB );
		float impulse = -massScale * ( Cdot + bias ) / invMass - impulseScale * joint->impulse;
		float newImpulse = b3MaxFloat( 0.0f, joint->impulse + impulse );
		impulse = newImpulse - joint->impulse;
		joint->impulse = newImpulse;

		b3RopeApply( base, &j, impulse, &vA, &wA, &vB, &wB );
	}

	// Bending, while taut: slack rope hangs as it likes.
	if ( b3RopeBendActive( joint ) )
	{
		if ( path.length < joint->length - B3_LINEAR_SLOP )
		{
			b3RopeBendRelease( joint );
		}
		else
		{
			b3RopeBend( base, joint, &path, stateA, stateB, context->h, true, &vA, &wA, &vB, &wB );
		}
	}

	// Relaxed (each substep's final velocities): the work friction does against the slip,
	// summed over the step.
	if ( useBias == false )
	{
		for ( int k = 1; k + 1 < joint->pointCount; ++k )
		{
			if ( joint->gripTighter[k] == 0.0f )
			{
				continue;
			}
			b3RopeJacobian jk = b3RopeJacobianOf( joint, &path, k + 1 );
			joint->slipEnergy[k] += b3AbsFloat( joint->frictionImpulse[k] * b3RopeRate( &jk, vA, wA, vB, wB ) );
		}
	}

	if ( stateA->flags & b3_dynamicFlag )
	{
		stateA->linearVelocity = vA;
		stateA->angularVelocity = wA;
	}

	if ( stateB->flags & b3_dynamicFlag )
	{
		stateB->linearVelocity = vB;
		stateB->angularVelocity = wB;
	}
}

void b3DrawRopeJoint( b3DebugDraw* draw, b3JointSim* base, b3WorldTransform transformA, b3WorldTransform transformB )
{
	B3_ASSERT( base->type == b3_ropeJoint );

	b3RopeJoint* joint = &base->ropeJoint;
	b3Pos previous = { 0 };
	for ( int i = 0; i < joint->pointCount; ++i )
	{
		b3WorldTransform xf = ( joint->bodyBMask >> i ) & 1u ? transformB : transformA;
		b3Pos p = b3TransformWorldPoint( xf, joint->localPoints[i] );
		if ( i > 0 )
		{
			draw->DrawSegmentFcn( previous, p, joint->impulse > 0.0f ? b3_colorOrange : b3_colorWhite, draw->context );
		}
		draw->DrawPointFcn( p, 4.0f, b3_colorWhite, draw->context );
		previous = p;
	}
}
