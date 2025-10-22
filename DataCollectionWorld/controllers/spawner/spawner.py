from controller import Supervisor
import random
import math

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

N_ROCKS = 120
N_SPHERES = 100
Z_FIXED = 0.02

XMIN, XMAX = -180.0, 180.0
YMIN, YMAX = -180.0, 180.0
AVOID_RADIUS = 3.0

root_children = supervisor.getRoot().getField("children")
spawned_position = []


def far_enough(x, z, positions, r=AVOID_RADIUS):
    rr = r * r
    for px, pz in positions:
        if (x - px) ** 2 + (z - pz) ** 2 < rr:
            return False
    return True


def spawn_item(name: str, color=None, radius=0.5):
    for _ in range(200):
        x = random.uniform(XMIN, XMAX)
        y = random.uniform(YMIN, YMAX)

        if far_enough(x, y, spawned_position):
            break
        else:
            x = random.uniform(XMIN, XMAX)
            y = random.uniform(YMIN, YMAX)

    if color is not None:
        node_str = f"""
        Solid {{
            translation {x:.5f} {y:.5f} {Z_FIXED:.5f}
            children [
                Shape {{
                    appearance PBRAppearance {{
                        baseColor {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}
                        roughness 0.5
                        metalness 0
                    }}
                    geometry Sphere {{
                        radius {radius:.3f}
                    }}
                }}
            ]
        }}"""
    else:
        node_str = f"""
        {name} {{
            translation {x:.5f} {y:.5f} {Z_FIXED:.5f}
            rotation 0 1 0 0
        }}"""
    root_children.importMFNodeFromString(-1, node_str)
    spawned_position.append((x, y))

def spawn_objectsOfInterest(name: str, count: int, color=None, radius=0.5):
    for _ in range(count):
        spawn_item(name, color, radius)

spawn_objectsOfInterest("Rock", N_ROCKS)
spawn_objectsOfInterest("Sphere", N_SPHERES, color=(1.0, 0.0, 0.0), radius=0.5)
