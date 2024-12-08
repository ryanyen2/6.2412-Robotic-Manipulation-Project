import os
import numpy as np
from pydrake.all import (
    Box, UnitInertia, SpatialInertia, RotationMatrix, ProximityProperties,
    AddContactMaterial, CoulombFriction, AddCompliantHydroelasticProperties,
    RigidTransform, RandomGenerator
)
from manipulation.scenarios import MakeManipulationStation

BRICK_LENGTH = 0.075  # x-dimension
BRICK_WIDTH = 0.05    # y-dimension
BRICK_HEIGHT = 0.05 
TABLE_HEIGHT = 0.22
rng = np.random.default_rng(172)
generator = RandomGenerator(rng.integers(0, 500))

def AddBrick(num_bricks, plant, generator, meshcat, name="brick"):
    brick_shape = Box(BRICK_LENGTH, BRICK_WIDTH, BRICK_HEIGHT)
    
    for i in range(num_bricks):
        brick_instance = plant.AddModelInstance(f"{name}{i}")
        mass = np.random.uniform(0.05, 0.45)

        # Random COM offset within 20% of the brick dimensions
        com_offset = np.array([
            np.random.uniform(-0.1 * BRICK_LENGTH, 0.1 * BRICK_LENGTH),
            np.random.uniform(-0.1 * BRICK_WIDTH, 0.1 * BRICK_WIDTH),
            np.random.uniform(-0.1 * BRICK_HEIGHT, 0.1 * BRICK_HEIGHT)
        ])

        G_SScm_E = UnitInertia.SolidBox(BRICK_LENGTH, BRICK_WIDTH, BRICK_HEIGHT)
        G_SP_E = G_SScm_E.ShiftFromCenterOfMass(-com_offset)

        I_SP_E = SpatialInertia(
            mass=mass,
            p_PScm_E=-com_offset,
            G_SP_E=G_SP_E
        )
        
        brick = plant.AddRigidBody(f"{name}{i}", brick_instance, I_SP_E)
        print(f"Added brick{i} with mass {mass} and com offset {-com_offset}")
        
        proximity_props = ProximityProperties()
        AddContactMaterial(
            friction=CoulombFriction(1.2, 1.0),
            properties=proximity_props
        )
        AddCompliantHydroelasticProperties(
            properties=proximity_props,
            hydroelastic_modulus=1.0e8,
            resolution_hint=0.002
        )

        plant.RegisterCollisionGeometry(
            brick,
            RigidTransform(),
            brick_shape,
            f"{name}{i}_collision",
            proximity_props
        )
        
        plant.RegisterVisualGeometry(
            brick,
            RigidTransform(),
            brick_shape,
            f"{name}{i}_visual",
            np.array([0.5, 0.5, 0.5, 0.8])
        )

def StationSetup(meshcat):
    model_directives = """
directives:
- add_directives:
    file: package://overhang/env.dmd.yaml
"""

    def prefinalize_callback(plant):
        AddBrick(4, plant, generator, meshcat)

    station = MakeManipulationStation(
        model_directives,
        package_xmls=[os.path.join(os.getcwd(), "models/package.xml")],
        time_step=0.001,
        prefinalize_callback=prefinalize_callback,
    )

    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")

    return station, plant, scene_graph