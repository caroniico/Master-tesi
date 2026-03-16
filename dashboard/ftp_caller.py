from ftplib import FTP
import tempfile
import tarfile
import zstandard as zstd
import os
import xarray as xr
import shutil
import matplotlib.pyplot as plt

HOST = "ocean.dmi.dk"
USER = "oceanftp"
PWD = "NYEflinte.stene"

ftp = FTP(HOST)
ftp.login(USER, PWD)
ftp.cwd("MBL/HIDRA3_training_data")

fname = "201912.tar.zst"  # example file, adjust as needed

tmp_tar_path = None
tmp_nc_path = None

try:
    # download monthly archive
    with tempfile.NamedTemporaryFile(suffix=".tar.zst", delete=False) as tmp:
        tmp_tar_path = tmp.name
        print("Downloading to:", tmp_tar_path)
        ftp.retrbinary(f"RETR {fname}", tmp.write, blocksize=1024 * 1024)

    print("Download completed.")

    with open(tmp_tar_path, "rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(compressed_file) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    if not member.name.endswith(".nc"):
                        continue

                    print("Opening first NetCDF found:", member.name)

                    extracted_file = tar.extractfile(member)
                    if extracted_file is None:
                        continue

                    # save only this one nc temporarily
                    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_nc:
                        tmp_nc_path = tmp_nc.name
                        shutil.copyfileobj(extracted_file, tmp_nc)

                    # inspect with xarray
                    with xr.open_dataset(tmp_nc_path) as ds:
                        varname = "var1"
                        print(ds)
                        print("\nVariables:")
                        print(list(ds.data_vars))
                        print("\nCoordinates:")
                        print(list(ds.coords))
                        da2d = ds[varname].isel(time=0, alt=0)  # example: first time step, surface level
                        
                        # PLot figure of the 2D variable at the first time step
                        plt.figure(figsize=(8, 6))
                        da2d.plot(x="lon", y="lat")
                        plt.title(f"{varname} at first time step")
                        plt.show()  


                    break  # only first .nc for now

finally:
    ftp.quit()

    if tmp_tar_path and os.path.exists(tmp_tar_path):
        os.remove(tmp_tar_path)
        print("Temporary tar deleted.")

    if tmp_nc_path and os.path.exists(tmp_nc_path):
        os.remove(tmp_nc_path)
        print("Temporary nc deleted.")

