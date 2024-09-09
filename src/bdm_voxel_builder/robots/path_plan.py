import click
import compas.geometry as cg
from compas.data import json_dump, json_load
from compas_fab.backends import RosClient

from bdm_voxel_builder import TEMP_DIR


def transform_frame(frame: cg.Frame):
    frame = cg.Frame(frame.point, frame.yaxis, frame.xaxis)
    S = cg.Scale.from_factors([0.001] * 3)
    Tl = cg.Translation.from_vector([0, 1000, 100])
    return frame.transformed(S * Tl)


def pointcloud_to_frames(pointcloud: cg.Pointcloud):
    yield from (cg.Frame(pt) for pt in pointcloud.points)


@click.command()
@click.argument("frames_file", type=click.Path(exists=True))
@click.option(
    "-i", "--ip", type=str, default="localhost", help="ROS Bridge IP to connect to"
)
@click.option(
    "-p", "--port", type=int, default=9090, help="ROS Bridge port to connect to"
)
def main(frames_file, ip, port):
    frames = json_load(frames_file or TEMP_DIR / "frames.json")
    with RosClient(ip, port) as ros_client:
        robot = ros_client.load_robot(load_geometry=True, precision=12)

        last_conf = robot.zero_configuration()
        confs = []
        for frame in pointcloud_to_frames(frames):
            frame = transform_frame(frame)
            conf = robot.inverse_kinematics(
                frame, start_configuration=last_conf, group=robot.main_group_name
            )
            last_conf = conf
            confs.append(conf)
        print(f"Length frames: {len(frames)}")
        print(f"Length confs: {len(confs)}")
        json_dump(confs, TEMP_DIR / "confs.json")


if __name__ == "__main__":
    main()
