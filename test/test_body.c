// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#include "test_macros.h"

#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "box3d/math_functions.h"

// Reach into internals to observe body extents and the dirty mass flag.
#include "body.h"
#include "physics_world.h"

#include <float.h>

// b3UpdateBodyMassData shifts each shape's inertia to the body center of mass with the parallel
// axis theorem. When shapes sit far from the body origin the shift term dwarfs the central inertia,
// so any error in the per shape framing blows up the tensor. Spheres make a clean oracle: the
// central inertia is isotropic and independent of placement, so the shift is the only thing tested.

static b3MassData SphereBodyMass( const b3Vec3* centers, int count, float radius, float density )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = density;

	for ( int i = 0; i < count; ++i )
	{
		b3Sphere sphere = { centers[i], radius };
		b3CreateSphereShape( bodyId, &shapeDef, &sphere );
	}

	b3Body_ApplyMassFromShapes( bodyId );
	b3MassData massData = b3Body_GetMassData( bodyId );

	b3DestroyWorld( worldId );
	return massData;
}

// One sphere far from the body origin. The center of mass lands on the sphere and the inertia about
// it must be the bare central inertia, with no trace of the offset.
static int FarSingleSphereMass( void )
{
	float radius = 0.5f;
	float density = 1.0f;
	b3Vec3 center = { 100.0f, -50.0f, 75.0f };
	b3MassData md = SphereBodyMass( &center, 1, radius, density );

	float mass = density * ( 4.0f / 3.0f ) * B3_PI * radius * radius * radius;
	float central = 0.4f * mass * radius * radius;

	ENSURE_SMALL( md.mass - mass, 1e-4f );

	ENSURE_SMALL( md.center.x - center.x, 1e-3f );
	ENSURE_SMALL( md.center.y - center.y, 1e-3f );
	ENSURE_SMALL( md.center.z - center.z, 1e-3f );

	ENSURE_SMALL( md.inertia.cx.x - central, 1e-3f );
	ENSURE_SMALL( md.inertia.cy.y - central, 1e-3f );
	ENSURE_SMALL( md.inertia.cz.z - central, 1e-3f );

	ENSURE_SMALL( md.inertia.cy.x, 1e-3f );
	ENSURE_SMALL( md.inertia.cz.x, 1e-3f );
	ENSURE_SMALL( md.inertia.cz.y, 1e-3f );

	return 0;
}

// Eight equal spheres on the corners of a cube, the whole cube parked far from the body origin.
// The center of mass is the cube center and the products of inertia cancel by symmetry, so the
// tensor stays diagonal no matter how far out the cube sits.
static int FarCubeSphereMass( void )
{
	float radius = 0.5f;
	float density = 1.0f;
	float h = 1.0f;
	b3Vec3 p = { 100.0f, 100.0f, 100.0f };

	b3Vec3 centers[8];
	int k = 0;
	for ( int sx = -1; sx <= 1; sx += 2 )
	{
		for ( int sy = -1; sy <= 1; sy += 2 )
		{
			for ( int sz = -1; sz <= 1; sz += 2 )
			{
				centers[k++] = (b3Vec3){ p.x + sx * h, p.y + sy * h, p.z + sz * h };
			}
		}
	}

	b3MassData md = SphereBodyMass( centers, 8, radius, density );

	float mass = density * ( 4.0f / 3.0f ) * B3_PI * radius * radius * radius;
	float totalMass = 8.0f * mass;

	// Per sphere central inertia summed, plus the parallel axis term for each corner offset
	// (dy^2 + dz^2) = (h^2 + h^2) about every axis.
	float diag = 8.0f * 0.4f * mass * radius * radius + 16.0f * mass * h * h;

	ENSURE_SMALL( md.mass - totalMass, 1e-3f );

	ENSURE_SMALL( md.center.x - p.x, 1e-2f );
	ENSURE_SMALL( md.center.y - p.y, 1e-2f );
	ENSURE_SMALL( md.center.z - p.z, 1e-2f );

	ENSURE_SMALL( md.inertia.cx.x - diag, 1e-2f );
	ENSURE_SMALL( md.inertia.cy.y - diag, 1e-2f );
	ENSURE_SMALL( md.inertia.cz.z - diag, 1e-2f );

	ENSURE_SMALL( md.inertia.cy.x, 1e-2f );
	ENSURE_SMALL( md.inertia.cz.x, 1e-2f );
	ENSURE_SMALL( md.inertia.cz.y, 1e-2f );

	return 0;
}

// Shapes added with updateBodyMass = false defer the mass update, which is also the only place
// body extents are computed. A body that reaches the solver with minExtent == B3_HUGE never passes
// the continuous collision gate. The dirty mass flag must track the deferral, and both
// ApplyMassFromShapes and SetMassData must leave finite extents behind.
static int DeferredMassExtents( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = 1.0f;
	shapeDef.updateBodyMass = false;

	b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.5f };

	// Deferred create leaves mass and extents untouched but marks the body dirty.
	b3BodyId applyId = b3CreateBody( worldId, &bodyDef );
	b3CreateSphereShape( applyId, &shapeDef, &sphere );

	b3World* world = b3GetWorld( applyId.world0 );
	b3Body* applyBody = b3GetBodyFullId( world, applyId );
	b3BodySim* applySim = b3GetBodySim( world, applyBody );

	ENSURE( ( applyBody->flags & b3_dirtyMass ) != 0 );
	ENSURE( applySim->minExtent == B3_HUGE );

	// ApplyMassFromShapes computes extents and clears the flag.
	b3Body_ApplyMassFromShapes( applyId );
	ENSURE( ( applyBody->flags & b3_dirtyMass ) == 0 );
	ENSURE( applySim->minExtent < B3_HUGE );

	// SetMassData alone must also produce finite extents and clear the flag (the issue #35 repro).
	b3BodyId massId = b3CreateBody( worldId, &bodyDef );
	b3CreateSphereShape( massId, &shapeDef, &sphere );

	b3Body* massBody = b3GetBodyFullId( world, massId );
	b3BodySim* massSim = b3GetBodySim( world, massBody );
	ENSURE( ( massBody->flags & b3_dirtyMass ) != 0 );

	b3Matrix3 inertia = { { 0.2f, 0.0f, 0.0f }, { 0.0f, 0.2f, 0.0f }, { 0.0f, 0.0f, 0.2f } };
	b3MassData massData = { 2.0f, { 0.0f, 0.0f, 0.0f }, inertia };
	b3Body_SetMassData( massId, massData );

	ENSURE( ( massBody->flags & b3_dirtyMass ) == 0 );
	ENSURE( massSim->minExtent < B3_HUGE );

	b3DestroyWorld( worldId );
	return 0;
}

// b3Body_SetMassData overrides the mass properties directly, bypassing the shapes. It must derive
// everything the solver reads from the supplied tensor: the inverse mass, the local inverse inertia,
// and the world inverse inertia rotated by the body orientation. Fixed rotation zeros the angular part.
// These tests drive it through the public getters, no shapes required.

// Diagonal inertia with inverses that are exact in float, so tolerances stay tight.
static const b3Matrix3 kDiagInertia = { { 2.0f, 0.0f, 0.0f }, { 0.0f, 4.0f, 0.0f }, { 0.0f, 0.0f, 8.0f } };

static int SetMassDataRoundTrip( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.position = (b3Pos){ 5.0f, -3.0f, 2.0f };
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );

	b3Vec3 center = { 0.1f, 0.2f, 0.3f };
	b3MassData massData = { 3.0f, center, kDiagInertia };
	b3Body_SetMassData( bodyId, massData );

	ENSURE_SMALL( b3Body_GetMass( bodyId ) - 3.0f, 1e-6f );
	ENSURE_SMALL( b3Body_GetInverseMass( bodyId ) - 1.0f / 3.0f, 1e-6f );

	b3MassData md = b3Body_GetMassData( bodyId );
	ENSURE_SMALL( md.mass - 3.0f, 1e-6f );
	ENSURE_SMALL( md.center.x - center.x, 1e-6f );
	ENSURE_SMALL( md.center.y - center.y, 1e-6f );
	ENSURE_SMALL( md.center.z - center.z, 1e-6f );
	ENSURE_SMALL( md.inertia.cx.x - 2.0f, 1e-6f );
	ENSURE_SMALL( md.inertia.cy.y - 4.0f, 1e-6f );
	ENSURE_SMALL( md.inertia.cz.z - 8.0f, 1e-6f );

	b3Vec3 localCenter = b3Body_GetLocalCenter( bodyId );
	ENSURE_SMALL( localCenter.x - center.x, 1e-6f );
	ENSURE_SMALL( localCenter.y - center.y, 1e-6f );
	ENSURE_SMALL( localCenter.z - center.z, 1e-6f );

	b3Matrix3 localInertia = b3Body_GetLocalRotationalInertia( bodyId );
	ENSURE_SMALL( localInertia.cx.x - 2.0f, 1e-6f );
	ENSURE_SMALL( localInertia.cy.y - 4.0f, 1e-6f );
	ENSURE_SMALL( localInertia.cz.z - 8.0f, 1e-6f );

	// Identity rotation, so the world inverse inertia is just the local inverse: diag(1/2, 1/4, 1/8).
	b3Matrix3 invWorld = b3Body_GetWorldInverseRotationalInertia( bodyId );
	ENSURE_SMALL( invWorld.cx.x - 0.5f, 1e-5f );
	ENSURE_SMALL( invWorld.cy.y - 0.25f, 1e-5f );
	ENSURE_SMALL( invWorld.cz.z - 0.125f, 1e-5f );
	ENSURE_SMALL( invWorld.cy.x, 1e-5f );
	ENSURE_SMALL( invWorld.cz.x, 1e-5f );
	ENSURE_SMALL( invWorld.cx.y, 1e-5f );
	ENSURE_SMALL( invWorld.cz.y, 1e-5f );
	ENSURE_SMALL( invWorld.cx.z, 1e-5f );
	ENSURE_SMALL( invWorld.cy.z, 1e-5f );

	// World center of mass is the body origin plus the local center under identity rotation.
	b3Pos worldCenter = b3Body_GetWorldCenter( bodyId );
	ENSURE_SMALL( worldCenter.x - ( 5.0f + center.x ), 1e-5f );
	ENSURE_SMALL( worldCenter.y - ( -3.0f + center.y ), 1e-5f );
	ENSURE_SMALL( worldCenter.z - ( 2.0f + center.z ), 1e-5f );

	b3DestroyWorld( worldId );
	return 0;
}

// The world inverse inertia must be the local inverse rotated into world space. A 90 degree turn about
// z swaps the x and y principal moments, so diag(1/2, 1/4, 1/8) becomes diag(1/4, 1/2, 1/8). This is the
// regression guard: before SetMassData rotated the tensor it left the world inverse inertia stale.
static int SetMassDataWorldInertiaRotated( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.rotation = b3MakeQuatFromAxisAngle( b3Vec3_axisZ, 0.5f * B3_PI );
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );

	b3MassData massData = { 1.0f, { 0.0f, 0.0f, 0.0f }, kDiagInertia };
	b3Body_SetMassData( bodyId, massData );

	// The local inertia is stored untouched by the world transform.
	b3Matrix3 localInertia = b3Body_GetLocalRotationalInertia( bodyId );
	ENSURE_SMALL( localInertia.cx.x - 2.0f, 1e-6f );
	ENSURE_SMALL( localInertia.cy.y - 4.0f, 1e-6f );
	ENSURE_SMALL( localInertia.cz.z - 8.0f, 1e-6f );

	b3Matrix3 invWorld = b3Body_GetWorldInverseRotationalInertia( bodyId );
	ENSURE_SMALL( invWorld.cx.x - 0.25f, 1e-4f );
	ENSURE_SMALL( invWorld.cy.y - 0.5f, 1e-4f );
	ENSURE_SMALL( invWorld.cz.z - 0.125f, 1e-4f );
	ENSURE_SMALL( invWorld.cy.x, 1e-4f );
	ENSURE_SMALL( invWorld.cz.x, 1e-4f );
	ENSURE_SMALL( invWorld.cx.y, 1e-4f );
	ENSURE_SMALL( invWorld.cz.y, 1e-4f );
	ENSURE_SMALL( invWorld.cx.z, 1e-4f );
	ENSURE_SMALL( invWorld.cy.z, 1e-4f );

	b3DestroyWorld( worldId );
	return 0;
}

// Fixed rotation must leave the mass intact but drive the whole angular inertia to zero, even when the
// caller hands in a real tensor.
static int SetMassDataFixedRotation( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.motionLocks.angularX = true;
	bodyDef.motionLocks.angularY = true;
	bodyDef.motionLocks.angularZ = true;
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );

	b3MassData massData = { 5.0f, { 0.0f, 0.0f, 0.0f }, kDiagInertia };
	b3Body_SetMassData( bodyId, massData );

	ENSURE_SMALL( b3Body_GetMass( bodyId ) - 5.0f, 1e-6f );
	ENSURE_SMALL( b3Body_GetInverseMass( bodyId ) - 0.2f, 1e-6f );

	b3Matrix3 localInertia = b3Body_GetLocalRotationalInertia( bodyId );
	ENSURE_SMALL( localInertia.cx.x, 1e-6f );
	ENSURE_SMALL( localInertia.cy.y, 1e-6f );
	ENSURE_SMALL( localInertia.cz.z, 1e-6f );

	b3Matrix3 invWorld = b3Body_GetWorldInverseRotationalInertia( bodyId );
	ENSURE_SMALL( invWorld.cx.x, 1e-6f );
	ENSURE_SMALL( invWorld.cy.y, 1e-6f );
	ENSURE_SMALL( invWorld.cz.z, 1e-6f );

	b3MassData md = b3Body_GetMassData( bodyId );
	ENSURE_SMALL( md.inertia.cx.x, 1e-6f );
	ENSURE_SMALL( md.inertia.cy.y, 1e-6f );
	ENSURE_SMALL( md.inertia.cz.z, 1e-6f );

	b3DestroyWorld( worldId );
	return 0;
}

// Zero mass and a zero tensor have zero determinant, so the inverses must collapse to zero rather than
// divide by it.
static int SetMassDataZeroMass( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );

	b3MassData massData = { 0.0f, { 0.0f, 0.0f, 0.0f }, b3Mat3_zero };
	b3Body_SetMassData( bodyId, massData );

	ENSURE_SMALL( b3Body_GetInverseMass( bodyId ), 1e-6f );

	b3Matrix3 localInertia = b3Body_GetLocalRotationalInertia( bodyId );
	ENSURE_SMALL( localInertia.cx.x, 1e-6f );
	ENSURE_SMALL( localInertia.cy.y, 1e-6f );
	ENSURE_SMALL( localInertia.cz.z, 1e-6f );

	b3Matrix3 invWorld = b3Body_GetWorldInverseRotationalInertia( bodyId );
	ENSURE_SMALL( invWorld.cx.x, 1e-6f );
	ENSURE_SMALL( invWorld.cy.y, 1e-6f );
	ENSURE_SMALL( invWorld.cz.z, 1e-6f );

	b3DestroyWorld( worldId );
	return 0;
}

// The stored linear velocity tracks the center of mass. Moving the center picks a different material
// point, so a spinning body must have its velocity re-referenced by omega x (newCenter - oldCenter),
// otherwise the mass edit silently injects or drains kinetic energy. Under identity rotation the world
// shift equals the supplied local center, keeping the expected values exact in float.
static int SetMassDataConsistentVelocity( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.position = (b3Pos){ 7.0f, 1.0f, -4.0f };
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );

	// Spin about the origin center, then shift the center of mass off the origin.
	b3Vec3 omega = { 1.0f, 2.0f, 4.0f };
	b3Body_SetLinearVelocity( bodyId, (b3Vec3){ 1.0f, -2.0f, 3.0f } );
	b3Body_SetAngularVelocity( bodyId, omega );

	b3Vec3 center = { 0.5f, 0.25f, 0.125f };
	b3MassData massData = { 3.0f, center, kDiagInertia };
	b3Body_SetMassData( bodyId, massData );

	// omega x center = ( 2*0.125 - 4*0.25, 4*0.5 - 1*0.125, 1*0.25 - 2*0.5 ) = ( -0.75, 1.875, -0.75 )
	b3Vec3 v = b3Body_GetLinearVelocity( bodyId );
	ENSURE_SMALL( v.x - ( 1.0f - 0.75f ), 1e-6f );
	ENSURE_SMALL( v.y - ( -2.0f + 1.875f ), 1e-6f );
	ENSURE_SMALL( v.z - ( 3.0f - 0.75f ), 1e-6f );

	// Only the reference point moved, the angular velocity is untouched.
	b3Vec3 w = b3Body_GetAngularVelocity( bodyId );
	ENSURE_SMALL( w.x - omega.x, 1e-6f );
	ENSURE_SMALL( w.y - omega.y, 1e-6f );
	ENSURE_SMALL( w.z - omega.z, 1e-6f );

	b3DestroyWorld( worldId );
	return 0;
}

// Extents bound the shapes about the center of mass, per axis. An offset shape must count its
// offset, not just its own size.
static int ShapeExtents( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = 1.0f;

	// Kinematic bodies measure from the body origin
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_kinematicBody;

	b3BodyId capsuleId = b3CreateBody( worldId, &bodyDef );
	b3Capsule capsule = { { -2.0f, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f }, 0.2f };
	b3CreateCapsuleShape( capsuleId, &shapeDef, &capsule );

	b3Vec3 maxExtent = b3Body_GetMaxExtent( capsuleId );
	ENSURE_SMALL( maxExtent.x - 2.2f, 1e-5f );
	ENSURE_SMALL( maxExtent.y - 0.2f, 1e-5f );
	ENSURE_SMALL( maxExtent.z - 0.2f, 1e-5f );
	ENSURE_SMALL( b3Body_GetMinExtent( capsuleId ) - 0.2f, 1e-5f );

	b3BodyId sphereId = b3CreateBody( worldId, &bodyDef );
	b3Sphere sphere = { { 1.0f, 2.0f, 3.0f }, 0.5f };
	b3CreateSphereShape( sphereId, &shapeDef, &sphere );

	maxExtent = b3Body_GetMaxExtent( sphereId );
	ENSURE_SMALL( maxExtent.x - 1.5f, 1e-5f );
	ENSURE_SMALL( maxExtent.y - 2.5f, 1e-5f );
	ENSURE_SMALL( maxExtent.z - 3.5f, 1e-5f );
	ENSURE_SMALL( b3Body_GetMinExtent( sphereId ) - 0.5f, 1e-5f );

	// Dynamic bodies measure from the center of mass. A light sphere hung off a cube pulls the
	// center toward it, so the far side of the sphere is the widest point.
	bodyDef.type = b3_dynamicBody;
	b3BodyId cubeId = b3CreateBody( worldId, &bodyDef );
	b3BoxHull cube = b3MakeCubeHull( 0.5f );
	b3CreateHullShape( cubeId, &shapeDef, &cube.base );
	b3Sphere offsetSphere = { { 1.0f, 0.0f, 0.0f }, 0.2f };
	b3CreateSphereShape( cubeId, &shapeDef, &offsetSphere );

	b3Vec3 localCenter = b3Body_GetLocalCenter( cubeId );
	ENSURE( 0.0f < localCenter.x && localCenter.x < 0.5f );

	maxExtent = b3Body_GetMaxExtent( cubeId );
	ENSURE_SMALL( maxExtent.x - ( 1.2f - localCenter.x ), 1e-5f );
	ENSURE_SMALL( maxExtent.y - 0.5f, 1e-5f );
	ENSURE_SMALL( maxExtent.z - 0.5f, 1e-5f );
	ENSURE_SMALL( b3Body_GetMinExtent( cubeId ) - 0.2f, 1e-5f );

	b3Vec3 originExtent = b3Body_GetMaxExtentOrigin( cubeId );
	ENSURE_SMALL( originExtent.x - 1.2f, 1e-5f );
	ENSURE_SMALL( originExtent.y - 0.5f, 1e-5f );
	ENSURE_SMALL( originExtent.z - 0.5f, 1e-5f );

	// A mesh has no mass, so the sphere alone places the center at x = 1 and the far edge of the
	// mesh at x = -3 is four units out.
	b3Vec3 vertices[6] = {
		{ -3.0f, 0.0f, -1.0f }, { -2.0f, 0.0f, 1.0f }, { -1.0f, 0.0f, -1.0f },
		{ 1.0f, 0.0f, -1.0f },	{ 2.0f, 0.0f, 1.0f },	{ 3.0f, 0.0f, -1.0f },
	};
	int32_t indices[6] = { 0, 1, 2, 3, 4, 5 };
	b3MeshDef meshDef = { 0 };
	meshDef.vertices = vertices;
	meshDef.stride = sizeof( b3Vec3 );
	meshDef.indices = indices;
	meshDef.vertexCount = 6;
	meshDef.triangleCount = 2;
	b3MeshData* mesh = b3CreateMesh( &meshDef, NULL, 0 );

	b3BodyId meshId = b3CreateBody( worldId, &bodyDef );
	b3CreateSphereShape( meshId, &shapeDef, &offsetSphere );
	b3CreateMeshShape( meshId, &shapeDef, mesh, (b3Vec3){ 1.0f, 1.0f, 1.0f } );

	localCenter = b3Body_GetLocalCenter( meshId );
	ENSURE_SMALL( localCenter.x - 1.0f, 1e-5f );

	maxExtent = b3Body_GetMaxExtent( meshId );
	ENSURE_SMALL( maxExtent.x - 4.0f, 1e-4f );
	ENSURE_SMALL( maxExtent.y - 0.2f, 1e-4f );
	ENSURE_SMALL( maxExtent.z - 1.0f, 1e-4f );
	ENSURE_SMALL( b3Body_GetMinExtent( meshId ) - 0.2f, 1e-5f );

	b3DestroyWorld( worldId );
	b3DestroyMesh( mesh );
	return 0;
}

// Move events are rebuilt each step for awake bodies only. A body that leaves the awake set by being
// disabled must drop its move event index, otherwise forcing it asleep after re-enabling writes
// through a stale index into another body's event or past the end of the array.
static int ForcedSleepAfterDisable( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyId bodyIds[3];
	for ( int i = 0; i < 3; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.position = (b3Pos){ 3.0f * i, 5.0f, 0.0f };
		bodyIds[i] = b3CreateBody( worldId, &bodyDef );
		b3ShapeDef shapeDef = b3DefaultShapeDef();
		b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.5f };
		b3CreateSphereShape( bodyIds[i], &shapeDef, &sphere );
	}

	// All three move, then two leave the awake set and the next step emits one move event
	b3World_Step( worldId, 1.0f / 60.0f, 4 );
	b3Body_Disable( bodyIds[0] );
	b3Body_Disable( bodyIds[1] );
	b3World_Step( worldId, 1.0f / 60.0f, 4 );

	b3World* world = b3GetWorldFromId( worldId );
	b3Body* body = world->bodies.data + ( bodyIds[1].index1 - 1 );
	ENSURE( body->bodyMoveIndex == B3_NULL_INDEX );

	b3Body_Enable( bodyIds[1] );
	b3Body_SetAwake( bodyIds[1], false );
	ENSURE( b3Body_IsAwake( bodyIds[1] ) == false );

	b3BodyEvents moveEvents = b3World_GetBodyEvents( worldId );
	ENSURE( moveEvents.moveCount == 1 );
	ENSURE( moveEvents.moveEvents[0].fellAsleep == false );

	b3DestroyWorld( worldId );
	return 0;
}

// A box resting on a box resting on the ground: one island of two bodies.
static b3WorldId CreateSleepStack( b3BodyId* bodyIds )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3BodyDef groundDef = b3DefaultBodyDef();
	b3BodyId groundId = b3CreateBody( worldId, &groundDef );
	b3BoxHull groundBox = b3MakeBoxHull( 10.0f, 0.5f, 10.0f );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3CreateHullShape( groundId, &shapeDef, &groundBox.base );

	b3BoxHull box = b3MakeBoxHull( 0.5f, 0.5f, 0.5f );
	for ( int i = 0; i < 2; ++i )
	{
		b3BodyDef bodyDef = b3DefaultBodyDef();
		bodyDef.type = b3_dynamicBody;
		bodyDef.position = (b3Pos){ 0.0f, 1.0f + i, 0.0f };
		bodyIds[i] = b3CreateBody( worldId, &bodyDef );
		b3CreateHullShape( bodyIds[i], &shapeDef, &box.base );
	}

	return worldId;
}

// A replica with the same bodies restarts its sleep timers at lateStep, as a peer that adopted the
// state late does. From syncStep on it takes the source's sleep times before each step, and the
// worlds must then hash equal. Writes the number of steps the replica fell asleep after the source.
static int SleepStepGap( int lateStep, int syncStep, int* gap )
{
	b3BodyId sourceIds[2], replicaIds[2];
	b3WorldId sourceId = CreateSleepStack( sourceIds );
	b3WorldId replicaId = CreateSleepStack( replicaIds );

	int sourceSleep = -1;
	int replicaSleep = -1;
	for ( int step = 0; step < 600 && ( sourceSleep < 0 || replicaSleep < 0 ); ++step )
	{
		for ( int i = 0; i < 2; ++i )
		{
			if ( step == lateStep )
			{
				b3Body_SetSleepTime( replicaIds[i], 0.0f );
			}

			if ( step >= syncStep )
			{
				b3Body_SetSleepTime( replicaIds[i], b3Body_GetSleepTime( sourceIds[i] ) );
			}
		}

		b3World_Step( sourceId, 1.0f / 60.0f, 4 );
		b3World_Step( replicaId, 1.0f / 60.0f, 4 );

		if ( step >= syncStep )
		{
			ENSURE( b3World_GetStateHash( sourceId ) == b3World_GetStateHash( replicaId ) );
		}

		if ( sourceSleep < 0 && b3Body_IsAwake( sourceIds[0] ) == false )
		{
			ENSURE( b3Body_IsAwake( sourceIds[1] ) == false );
			sourceSleep = step;
		}

		if ( replicaSleep < 0 && b3Body_IsAwake( replicaIds[0] ) == false )
		{
			replicaSleep = step;
		}
	}

	ENSURE( sourceSleep > lateStep && replicaSleep > lateStep );
	*gap = replicaSleep - sourceSleep;

	b3DestroyWorld( sourceId );
	b3DestroyWorld( replicaId );
	return 0;
}

// Copying sleep times makes islands fall asleep on the same step, also when the copy starts a few
// steps before the threshold.
static int SleepTimeSync( void )
{
	int gap = 0;
	ENSURE( SleepStepGap( 15, INT32_MAX, &gap ) == 0 );
	ENSURE( gap >= 10 );

	ENSURE( SleepStepGap( 15, 16, &gap ) == 0 );
	ENSURE( gap == 0 );

	int thresholdStep = (int)( B3_TIME_TO_SLEEP * 60.0f );
	ENSURE( SleepStepGap( 15, thresholdStep - 3, &gap ) == 0 );
	ENSURE( gap == 0 );
	return 0;
}

// A sleeping body ignores sleep times at the threshold and wakes its island below it. Static bodies
// ignore sleep times.
static int SleepTimeOnSleepingBody( void )
{
	b3BodyId bodyIds[2];
	b3WorldId worldId = CreateSleepStack( bodyIds );

	for ( int step = 0; step < 600 && b3Body_IsAwake( bodyIds[0] ); ++step )
	{
		b3World_Step( worldId, 1.0f / 60.0f, 4 );
	}

	ENSURE( b3Body_IsAwake( bodyIds[0] ) == false );
	ENSURE( b3Body_GetSleepTime( bodyIds[0] ) >= B3_TIME_TO_SLEEP );

	b3Body_SetSleepTime( bodyIds[0], B3_TIME_TO_SLEEP );
	ENSURE( b3Body_IsAwake( bodyIds[0] ) == false );

	b3Body_SetSleepTime( bodyIds[0], 0.1f );
	ENSURE( b3Body_IsAwake( bodyIds[0] ) && b3Body_IsAwake( bodyIds[1] ) );
	ENSURE( b3Body_GetSleepTime( bodyIds[0] ) == 0.1f );
	ENSURE( b3Body_GetSleepTime( bodyIds[1] ) == 0.0f );

	b3BodyDef staticDef = b3DefaultBodyDef();
	staticDef.position = (b3Pos){ 20.0f, 0.0f, 0.0f };
	b3BodyId staticId = b3CreateBody( worldId, &staticDef );
	b3Body_SetSleepTime( staticId, 0.2f );
	ENSURE( b3Body_GetSleepTime( staticId ) == 0.0f );

	b3DestroyWorld( worldId );
	return 0;
}

int BodyTest( void )
{
	RUN_SUBTEST( SleepTimeSync );
	RUN_SUBTEST( SleepTimeOnSleepingBody );
	RUN_SUBTEST( ForcedSleepAfterDisable );
	RUN_SUBTEST( FarSingleSphereMass );
	RUN_SUBTEST( FarCubeSphereMass );
	RUN_SUBTEST( DeferredMassExtents );
	RUN_SUBTEST( SetMassDataRoundTrip );
	RUN_SUBTEST( SetMassDataWorldInertiaRotated );
	RUN_SUBTEST( SetMassDataFixedRotation );
	RUN_SUBTEST( SetMassDataZeroMass );
	RUN_SUBTEST( SetMassDataConsistentVelocity );
	RUN_SUBTEST( ShapeExtents );
	return 0;
}
