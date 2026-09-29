// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "box3d/box3d.h"

typedef struct b3World b3World;

// Mutation marks from the public API. Called before the mutation so the pre-mutation topology is
// captured, the capture expands the same roots again against the post-mutation topology.
void b3DeltaMarkBody( b3World* world, int bodyId );
void b3DeltaMarkShape( b3World* world, int shapeId );
void b3DeltaMarkJoint( b3World* world, int jointId );

// Releases the hull references the tracker holds and unhooks it from the world. The tracker memory
// stays valid until b3DestroyWorldDelta.
void b3DeltaDetachWorld( b3World* world );

// Debug: log the undo bytes per table recorded by the last capture
void b3DeltaPrintTableBytes( const b3WorldDelta* delta );
