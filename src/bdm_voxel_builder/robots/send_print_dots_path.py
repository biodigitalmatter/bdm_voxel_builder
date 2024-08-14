"""
from gh
"""

# from compas import json_dump, json_load
import json

import compas_rrc as rrc
from bdm_voxel_builder.helpers.file import get_nth_newest_file_in_folder
from compas.geometry import Frame, Point, Pointcloud
from compas.geometry.transformation import Transformation as T
from compas_rhino.conversions import plane_to_compas_frame

# def load_pointcloud(file=None):
#     # TODO
#     pointcloud = None
#     return pointcloud


def send_program_dots(
    planes: list[Frame],
    client=None,
    speed=100,
    movel=True,
    zone="fine",
    print_IO="True",
    tool_name=None,
    wobj_name=None,
    print_IO_name="LOCAL_IO_5",
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
            digital_value = print_IO[i] if isinstance(zone, list) else print_IO
            digital_value = 1 if digital_value in (True, 1) else 0
            motion_type = movel[i] if isinstance(movel, list) else movel

            frame = plane_to_compas_frame(plane)
            print(f"send move to frame :: {i}")

            if digital_value == 0:
                client.send(rrc.MoveToFrame(frame, sp, zo, motion_type))
            else:
                extrude_and_wait(frame, sp, zo, motion_type, wait_time=2)


def extrude_and_wait(frame, sp, zo, motion_type, wait_time=2, wait_time_after=0):
    cmds = [
        rrc.MoveToFrame(frame, sp, zo, motion_type),
        rrc.SetDigital(io_name=print_IO_NAME, value=1),
        rrc.WaitTime(wait_time),
        rrc.SetDigital(io_name=print_IO_NAME, value=0),
        rrc.WaitTime(wait_time_after),
    ]
    [client.send(cmd) for cmd in cmds]


def extrude_with_z_hop(
    frame, sp, zo, motion_type, wait_time=2, wait_time_after=0, z_hop=15
):
    to_z_frame = Frame([0, 0, z_hop], [1, 0, 0], [0, 1, 0])
    move_z = T.from_frame(to_z_frame)
    frame_z_hop = Frame.transform(move_z)

    cmds = [
        rrc.MoveToFrame(frame_z_hop, sp, zo, motion_type),
        rrc.MoveToFrame(frame, sp, zo, motion_type),
        rrc.SetDigital(io_name=print_IO_NAME, value=1),
        rrc.WaitTime(wait_time),
        rrc.SetDigital(io_name=print_IO_NAME, value=0),
        rrc.WaitTime(wait_time_after),
        rrc.MoveToFrame(frame_z_hop, sp, zo, motion_type),
    ]
    [client.send(cmd) for cmd in cmds]


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

    print_IO_NAME = "LOCAL_IO_5"

    tool_name = None
    wobj_name = None

    send_program_dots(
        frames,
        client,
        speed,
        movel,
        zone,
        print_IO,
        tool_name,
        wobj_name,
    )
