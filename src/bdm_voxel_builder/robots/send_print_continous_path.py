"""
from gh
"""

# from compas import json_dump, json_load
import json

import compas_rrc as rrc
from bdm_voxel_builder.helpers.file import get_nth_newest_file_in_folder
from compas.geometry import Frame, Point, Pointcloud
from compas_rhino.conversions import plane_to_compas_frame

# def load_pointcloud(file=None):
#     # TODO
#     pointcloud = None
#     return pointcloud


def send_program(
    planes: list[Frame],
    client=None,
    speed=100,
    movel=True,
    zone="fine",
    tool_name=None,
    wobj_name=None,
):
    run = True

    # planes
    planes = planes

    if tool_name is None:
        tool_name = "tool0"
    if wobj_name is None:
        wobj_name = "wobj0"
    if speed is None:
        speed = 100

    if isinstance(movel, list):
        motion_type = []
        for linear in movel:
            if linear:
                motion_type.append(rrc.Motion.LINEAR)
            else:
                motion_type.append(rrc.Motion.JOINT)
    elif movel:
        motion_type = rrc.Motion.LINEAR
    elif not movel:
        motion_type = rrc.Motion.JOINT

    # print("Z50" in rrc.Zone.__dict__)
    # print(50 in rrc.Zone.__dict__.values())
    if zone and not isinstance(zone, list):
        if zone in rrc.Zone.__dict__:
            zone = rrc.Zone.__dict__[zone]
        elif zone in rrc.Zone.__dict__.values():
            pass  # do nothing
        else:
            raise ValueError("Given zone is not valid, see compas_rrc.Zone")
    elif zone and isinstance(zone, list):
        pass
        # for z in zone:
        #     if z in rrc.Zone.__dict__:
        #         z = rrc.Zone.__dict__[z]
        #     elif z in rrc.Zone.__dict__.values():
        #         pass  # do nothing
        #     else:
        #         raise ValueError("Given zone is not valid, see compas_rrc.Zone")
        #     pass
    else:
        zone = rrc.Zone.FINE

    if not client:
        raise Exception("No client, click to connect to ROS first.")
    elif not client.ros.is_connected:
        raise Exception("Not connected to ROS")

    if run:
        client.send(rrc.SetTool(tool_name))
        client.send(rrc.SetWorkObject(wobj_name))
        client.send(rrc.SetAcceleration(100, 100))
        client.send(rrc.SetMaxSpeed(100, 150))
        for i, plane in enumerate(planes):
            sp = speed[i] if isinstance(speed, list) else speed
            zo = zone[i] if isinstance(zone, list) else zone
            motion_type = movel[i] if isinstance(movel, list) else movel

            frame = plane_to_compas_frame(plane)
            print(f"send move to frame :: {i}")

            client.send(rrc.MoveToFrame(frame, sp, zo, motion_type))


# ??? does move_to_frame
# orients frames to workobject or global space ????


if __name__ == "__main__":
    # get client object
    client = None  # TODO

    folder_path = ""
    file = get_nth_newest_file_in_folder(folder_path)
    print(f"import file: {file}")

    data = json.load(file)
    frames = data["frames"]
    print_IO = data["print_IO"]
    speed = data["speed"]
    zone = data["zone"]
    movel = data["movel"]

    tool_name = None
    wobj_name = None

    send_program(
        frames,
        client,
        speed,
        movel,
        zone,
        tool_name,
        wobj_name,
    )
