import pathlib
import glob
import multiprocessing
import yaml
from skimage.morphology import white_tophat, black_tophat, disk
import numpy as np
import tifffile
import bigfish.stack as stack
import bigfish.detection as detection
from cellpose import models


# calculate psf (thank you MK), with edit for consistent nomenclature
def calculate_psf(voxel_size_z, voxel_size_yx, Ex, Em, NA, RI, microscope):
    """
    Use the formula implemented in Matlab version (sigma_PSF_BoZhang_v1)
    to calculate the theoretical PSF size.
    """
    if microscope == "widefield":
        psf_yx = 0.225 * Em / NA
        psf_z = 0.78 * RI * Em / (NA ** 2)
    elif microscope in ("confocal", "nipkow"):
        psf_yx = 0.225 / NA * Ex * Em / np.sqrt(Ex ** 2 + Em ** 2)
        psf_z = 0.78 * RI / NA ** 2 * Ex * Em / np.sqrt(Ex ** 2 + Em ** 2)
    else:
        # Unrecognised microscope
        raise Exception(
            "Unrecognised microscope argument for function calculate_psf()."
        )
    return psf_z, psf_yx


# subtract background
def subtract_background(image, radius=5, light_bg=False):
    # you can also use 'ball' here to get a slightly smoother result at the
    # cost of increased computing time
    str_el = disk(radius)
    if light_bg:
        return black_tophat(image, str_el)
    else:
        return white_tophat(image, str_el)


def image_processing_function(image_path, config):

    image_channels = config["channels"]
    voxel_size_yx = config["voxel_size_yx"]
    voxel_size_z = config["voxel_size_z"]

    # Read the image into a numpy array of format ZCYX
    image = tifffile.imread(image_path)

    # segment with cellpose
    if config["cp_search_string"] in image_path:
        seg_img = np.max(image[:, config["dapi_ch"], :, :], 0)
        seg_img = np.clip(seg_img, 0, config["cp_clip"])
    else:
        seg_img = np.max(image[:, 1, :, :], 0)
    model = models.Cellpose(gpu=config["gpu"], model_type="cyto")
    channels = [0, 0]  # greyscale segmentation
    masks = model.eval(
        seg_img,
        channels=channels,
        diameter=config["diameter"],
        do_3D=config["do_3D"],
    )[0]

    # Calculate PSF
    psf_z, psf_yx = calculate_psf(
        voxel_size_z,
        voxel_size_yx,
        config["ex"],
        config["em"],
        config["NA"],
        config["RI"],
        config["microscope"],
    )
    sigma = detection.get_sigma(voxel_size_z, voxel_size_yx, psf_z, psf_yx)

    for image_channel in image_channels:
        # detect spots
        rna = image[:, image_channel, :, :]
        rna = rna[:, :512, :512]  # TODO: cropping?
        # subtract background
        rna_no_bg = []
        for z in rna:
            z_no_bg = subtract_background(z)
            rna_no_bg.append(z_no_bg)
        rna = np.array(rna_no_bg)

        # LoG filter
        rna_log = stack.log_filter(rna, sigma)

        # local maximum detection
        mask = detection.local_maximum_detection(rna_log, min_distance=sigma)

        # thresholding
        if config["thresh_search_str1"] in image_path:
            threshold = config["thresh1"]
        else:
            threshold = config["thresh2"]
        spots, _ = detection.spots_thresholding(rna_log, mask, threshold)

        # detect and decompose clusters
        spots_post_decomposition = detection.decompose_cluster(
            rna,
            spots,
            voxel_size_z,
            voxel_size_yx,
            psf_z,
            psf_yx,
            alpha=0.7,  # alpha impacts the number of spots per cluster
            beta=1,  # beta impacts the number of detected clusters
        )[0]

        # separate spots from clusters
        spots_post_clustering, foci = detection.detect_foci(
            spots_post_decomposition,
            voxel_size_z,
            voxel_size_yx,
            config["radius"],
            config["nb_min_spots"],
        )

        # extract cell level results
        image_contrasted = stack.rescale(rna, channel_to_stretch=0)
        image_contrasted = stack.maximum_projection(image_contrasted)
        rna_mip = stack.maximum_projection(rna)

        fov_results = stack.extract_cell(
            cell_label=masks.astype(np.int64),
            ndim=3,
            rna_coord=spots_post_clustering,
            others_coord={"foci": foci},
            image=image_contrasted,
            others_image={"smfish": rna_mip},
        )

        # save results
        for i, cell_results in enumerate(fov_results):
            output_path = pathlib.Path(config["output_dir"]).joinpath(
                f"{pathlib.Path(image_path).name}_ch{image_channel+1}_results_cell_{i}.npz"
            )
            stack.save_cell_extracted(cell_results, str(output_path))


def image_processing_function_wrapper(args):
    return image_processing_function(*args)


def main():
    # Main entry point of the program
    # load config file
    with open("smFISH_analysis_config.yaml") as fi:
        config = yaml.load(fi, Loader=yaml.Loader)
    # Check if output dir exists; try to create it if not
    pathlib.Path(config["output_dir"]).mkdir(exist_ok=True)
    # multiprocessing.set_start_method("spawn") # TODO: revise
    with multiprocessing.Pool(config["number_of_workers"]) as p:
        image_paths = glob.glob(config["input_pattern"])
        process_data = [(image_path, config) for image_path in image_paths]
        p.map(image_processing_function_wrapper, process_data)


if __name__ == "__main__":
    main()
