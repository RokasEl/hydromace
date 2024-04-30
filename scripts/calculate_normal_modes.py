import ase.io as aio
import typer
from mace.calculators import mace_off
from wfl.autoparallelize import AutoparaInfo
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate import normal_modes as nm

app = typer.Typer()


@app.command()
def main(data_path: str, save_path: str, index: str = ":"):
    data = aio.read(data_path, index=index, format="extxyz")
    configset = ConfigSet(data)
    outputspec = OutputSpec(save_path)
    calc = (mace_off, ["medium"], {"default_dtype": "float64", "device": "cuda"})

    nm.generate_normal_modes_parallel_atoms(
        inputs=configset,
        outputs=outputspec,
        calculator=calc,
        prop_prefix="mace_",
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess=1),
    )


if __name__ == "__main__":
    app()
