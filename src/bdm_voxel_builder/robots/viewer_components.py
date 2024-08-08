import click
from compas.data import json_load
from compas_fab.backends import RosClient
from compas_robots.scene import RobotModelObject
from compas_viewer import Viewer
from compas_viewer.components import Slider, Treeform


@click.command()
@click.argument("confs", type=click.Path(exists=True))
@click.option(
    "-i", "--ip", type=str, default="localhost", help="ROS Bridge IP to connect to"
)
@click.option(
    "-p", "--port", type=int, default=9090, help="ROS Bridge port to connect to"
)
def main(confs, ip, port):
    viewer = Viewer(rendermode="lighted")
    confs = json_load(confs)

    with RosClient(ip, port) as ros_client:
        robot = ros_client.load_robot(load_geometry=True, precision=12)

        robot_object: RobotModelObject = viewer.scene.add(robot.model)

        robot_object.update_joints(robot.zero_configuration())

        def slider_func(index: int):
            robot_object.update_joints(robot_object.configurations[index])

        slider = Slider(
            name="Configuration",
            min_val=0,
            max_val=len(confs) - 1,
            action=slider_func,
        )

        viewer.layout.sidedock.add_element(slider)

        treeform = Treeform(
            viewer.scene, {"Name": (lambda o: o.name), "Object": (lambda o: o)}
        )

        viewer.layout.sidedock.add_element(treeform)

        viewer.show()


if __name__ == "__main__":
    main()
