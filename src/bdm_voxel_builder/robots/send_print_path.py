import compas.geometry as cg
import compas_rrc as rrc

# from bdm_voxel_builder.helpers.file import get_nth_newest_file_in_folder
from compas import json_load
from compas_fab.backends.ros import RosClient
from compas_rrc import AbbClient

from bdm_voxel_builder import DATA_DIR
from bdm_voxel_builder.helpers.file import get_nth_newest_file_in_folder

RUN_EXTRUDER_DO = "Local_IO_0_DO5"
DIR_EXTRUDER_DO = "Local_IO_0_DO6"
# RUN_PRESSURE = "Local_IO_0_DO2"


def coerce_zone(zone):
    """Returns a compas_rrc.Zone value given string or number"""
    if zone in rrc.Zone.__dict__:
        return rrc.Zone.__dict__[zone]
    elif zone in rrc.Zone.__dict__.values():
        return zone
    else:
        raise ValueError("Given zone is not valid, see compas_rrc.Zone")


def send_program_dots(
    frames: list[cg.Frame],
    speed: float = 100,
    movel: bool = True,
    zone: rrc.Zone = 5,
    print_IO=0,
    dir_IO=0,
    wait_time=1,
    tool_name=None,
    wobj_name=None,
    dot_print_style=True,
):
    run = True

    # frames
    frames = frames

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

    if run:
        with RosClient() as ros_client:
            print(ros_client)
            client = AbbClient(ros=ros_client)
            client.send(rrc.SetTool(tool_name))
            client.send(rrc.SetWorkObject(wobj_name))
            client.send(rrc.SetAcceleration(100, 100))
            client.send(rrc.SetMaxSpeed(100, 150))

            # first_frame = frames[0].translated(cg.Vector.Zaxis() * 250)

            client.send(
                rrc.MoveToJoints(
                    [90, 30, 15, 45, -45, -45],
                    0,
                    speed=100,
                    zone=rrc.Zone.FINE,
                )
            )

            for i, plane in enumerate(frames):
                sp = speed[i] if isinstance(speed, list) else speed
                zo = zone[i] if isinstance(zone, list) else zone
                wait_i = wait_time[i] if isinstance(wait_time, list) else wait_time
                IO_5 = print_IO[i] if isinstance(print_IO, list) else print_IO
                IO_6 = dir_IO[i] if isinstance(dir_IO, list) else dir_IO
                motion_type = movel[i] if isinstance(movel, list) else movel
                frame = plane
                print(f"send :: {i}")
                print(f"next frame: {frame}")
                print(f"io: {IO_5}")

                # if i % 50 == 0:
                #     client.send_and_wait(rrc.PrintText(f"Sent {i} planes"))
                client.send(rrc.PrintText(f"no.: {i}"))

                if dot_print_style:  # dot style print with z hop
                    if IO_5 == 0:
                        client.send(rrc.MoveToFrame(frame, sp, 5, motion_type))
                    elif IO_5 == 1 and IO_6 == 1:
                        client.send(rrc.MoveToFrame(frame, sp, 5, motion_type))
                        client.send(rrc.SetDigital(io_name=RUN_EXTRUDER_DO, value=IO_5))
                        client.send(rrc.SetDigital(io_name=DIR_EXTRUDER_DO, value=IO_6))
                    elif IO_5 == 1 and IO_6 == 0:
                        extrude_with_z_hop(
                            client,
                            frame,
                            sp,
                            motion_type,
                            wait_time=wait_i,
                            wait_time_after=0,
                            z_hop=45,
                        )

                else:  # continuous print path
                    client.send(rrc.MoveToFrame(frame, sp, zo, motion_type))
                    client.send(rrc.SetDigital(io_name=RUN_EXTRUDER_DO, value=IO_5))
                    # client.send(rrc.SetDigital(io_name=DIR_EXTRUDER_DO, value=IO_6))

            # client.send(rrc.SetDigital(io_name=RUN_PRESSURE, value=0))


def send_test_program(
    frames: list[cg.Frame],
    tool_name=None,
    wobj_name=None,
):
    run = True

    if run:
        with RosClient() as ros_client:
            print(ros_client)
            client = AbbClient(ros=ros_client)
            client.send(
                rrc.MoveToJoints(
                    [90, 30, 15, 45, -45, -45],
                    0,
                    speed=100,
                    zone=rrc.Zone.FINE,
                )
            )
            client.send(rrc.SetTool(tool_name))
            client.send(rrc.SetWorkObject(wobj_name))
            client.send(rrc.SetAcceleration(100, 100))
            client.send(rrc.SetMaxSpeed(100, 150))

            for i, plane in enumerate(frames):
                print(f"send :: {i}")
                print(f"next frame: {plane}")
                client.send(rrc.MoveToFrame(plane, 100, rrc.Zone.Z5, rrc.Motion.LINEAR))

            # client.send(rrc.SetDigital(io_name=RUN_PRESSURE, value=0))


def extrude_and_wait(
    client: rrc.AbbClient,
    frame: cg.Frame,
    speed: float,
    motion_type: rrc.Motion,
    zone: rrc.Zone = rrc.Zone.FINE,
    wait_time: float = 0.5,
    wait_time_after: float = 0.0,
):
    client.send(
        rrc.MoveToFrame(frame, speed, zone, motion_type),
    )
    client.send(rrc.SetDigital(io_name=RUN_EXTRUDER_DO, value=1))
    client.send(rrc.WaitTime(wait_time))
    client.send(
        rrc.SetDigital(io_name=RUN_EXTRUDER_DO, value=0),
    )
    client.send(
        rrc.WaitTime(wait_time_after),
    )


def extrude_with_z_hop(
    client: rrc.AbbClient,
    frame: cg.Frame,
    speed: float,
    motion_type: rrc.Motion,
    zone: rrc.Zone = rrc.Zone.Z5,
    wait_time: float = 0.5,
    wait_time_after: float = 0.0,
    z_hop: float = 15.0,
):
    z_hop_vector = frame.zaxis * z_hop

    # check if z_hop_vector is pointing in the right direction
    dot_product_with_z = frame.zaxis.dot(cg.Vector.Zaxis())
    if dot_product_with_z < 0:
        z_hop_vector.invert()

    frame_above = frame.transformed(cg.Translation.from_vector(z_hop_vector))

    client.send(rrc.MoveToFrame(frame_above, speed, zone, motion_type))
    client.send(rrc.MoveToFrame(frame, 25, zone, motion_type=motion_type))
    client.send(rrc.SetDigital(io_name=RUN_EXTRUDER_DO, value=1))
    client.send(rrc.WaitTime(wait_time))
    client.send(rrc.SetDigital(io_name=RUN_EXTRUDER_DO, value=0))
    if wait_time_after > 0:
        client.send(rrc.WaitTime(wait_time_after))
    client.send(rrc.MoveToFrame(frame_above, 25, zone, motion_type))


if __name__ == "__main__":
    # get client object
    # file_name = "fab_data_cylinder_2_lutum.json"
    # filepath = DATA_DIR / "live" / "fab_data" / file_name

    filepath = get_nth_newest_file_in_folder(DATA_DIR / "live" / "fab_data")
    print(f"import file: {filepath}")

    data = json_load(filepath)
    frames = data["frames"]
    print_IO = data["print_IO"]
    dir_IO = data["dir_IO"]
    speed = data["speed"]
    zone = data["zone"]
    movel = data["movel"]
    wait_time = data["wait_time"]
    zone = 5

    tool_name = "t_lutum_2"
    wobj_name = "wobj_bhg"

    dot_print_style = True

    send_program_dots(
        frames=frames,
        speed=100,
        movel=True,
        zone=None,
        print_IO=print_IO,
        dir_IO=0,
        wait_time=wait_time,
        tool_name=tool_name,
        wobj_name=wobj_name,
        dot_print_style=dot_print_style,
    )

    # send_test_program(frames, tool_name, wobj_name)
