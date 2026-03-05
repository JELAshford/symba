"""Send numpy arrays from a "Simulation" to local webpage for visualisation"""

from flask import Flask, render_template, Response
from tempfile import TemporaryDirectory
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import shutil
import time


class Simulation:
    def __init__(self):
        self.img_buffer = BytesIO()
        self.frame_index = 0

    def step(self) -> np.ndarray:
        raise NotImplementedError()

    def emit_buffer(
        self, target_fps: int = 60, img_format: str = "PNG", save_dir: Path = None
    ):
        # Clean up previous image outputs
        if save_dir:
            if save_dir.exists():
                shutil.rmtree(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)

        for output_array in self.step():
            img = Image.fromarray(output_array, "RGB")
            self.img_buffer.seek(0)
            self.img_buffer.truncate()
            img.save(self.img_buffer, format=img_format)
            self.img_buffer.seek(0)
            # also save to file
            if save_dir:
                img.save(save_dir / f"frame{self.frame_index}.{img_format.lower()}")
                self.frame_index += 1

            yield (
                b"--frame\r\nContent-Type: image/"
                + img_format.lower().encode()
                + b"\r\n\r\n"
                + self.img_buffer.read()
                + b"\r\n"
            )
            # TODO; Use time differences to make this accurate
            time.sleep(1 / target_fps)


class DisplayApp:
    def __init__(
        self,
        simulation_instance: Simulation,
        site_path: Path = TemporaryDirectory(),
        host: str = "0.0.0.0",
        port: int = 8080,
        debug: bool = True,
        target_fps: int = 60,
        save_dir: Path = None,
    ):
        self.simulation_instance = simulation_instance
        self.app_params = dict(host=host, port=port, debug=debug)
        self.site_path = Path(site_path)
        self.app = Flask(
            __name__,
            template_folder=self.site_path.absolute(),
            static_folder=self.site_path.absolute(),
        )
        self.target_fps = target_fps
        self.save_dir = save_dir

    @staticmethod
    def write_file(path: Path, content: str):
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            path.absolute().touch()
        with open(path, "w") as file:
            file.write(content)

    def setup_site(self):
        self.write_file(
            self.site_path / "style.css",
            content="""
img {
  width: auto;
  height: 100vh;
  image-rendering: pixelated;
  image-rendering: -moz-crisp-edges;
  image-rendering: crisp-edges;
}

body {
  margin: 0;
  padding: 0;
  background: rgb(0, 0, 0, 0);
  display: flex;
}
""",
        )

        self.write_file(
            self.site_path / "index.html",
            content=f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Array Display</title>
    <link
      rel="stylesheet"
      href="{self.site_path / 'style.css'}"
    />
  </head>
  <body>
    <div>
      <img src="/video_feed" />
    </div>
  </body>
</html>
""",
        )

    def setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self.simulation_instance.emit_buffer(
                    target_fps=self.target_fps,
                    save_dir=self.save_dir,
                ),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def run(self):
        self.setup_site()
        self.setup_routes()
        self.app.run(**self.app_params)
