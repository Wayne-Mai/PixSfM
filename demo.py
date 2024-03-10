import tqdm, tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive, visualization
from hloc.visualization import plot_images, read_image
from hloc.utils.viz_3d import init_figure, plot_points, plot_reconstruction, plot_camera_colmap

from pixsfm.util.visualize import init_image, plot_points2D
from pixsfm.refine_hloc import PixSfM
# from pixsfm import ostream_redirect

# redirect the C++ outputs to notebook cells
# cpp_out = ostream_redirect(stderr=True, stdout=True)



images = Path('/home/maij/text3d/sparf/third_party/pixel-perfect-sfm/datasets/sacre_coeur')
outputs = Path('/home/maij/text3d/sparf/third_party/pixel-perfect-sfm/outputs/demo/')
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'
raw_dir = outputs / "raw"
ref_dir = outputs / "ref"

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']


references = [str(p.relative_to(images)) for p in (images / 'mapping/').iterdir()]
print(len(references), "mapping images")

extract_features.main(feature_conf, images, image_list=references, feature_path=features)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

from pixsfm.refine_hloc import PixSfM
from pixsfm.util.hloc import read_keypoints_hloc
# features = outputs / 'features.h5'
# matches = outputs / 'matches.h5'
print(outputs)

# with open(sfm_pairs, "r") as f:
#     contents = f.read()
#     print(contents)

# from pixsfm.util.hloc import read_image_pairs
# pairs = read_image_pairs(sfm_pairs)
# print(pairs)

refiner = PixSfM()

print(refiner.test_read(sfm_pairs))

# def refine_keypoints(
#         self,
#         output_path: Path,
#         features_path: Path,
#         image_dir: Path,
#         pairs_path: Path,
#         matches_path: Path,
#         cache_path: Optional[Path] = None,
#         feature_manager: Optional[FeatureManager] = None):


keypoints_new, _, _ = refiner.refine_keypoints(
    outputs / 'features_refined.h5',
    features,
    images,
    sfm_pairs,
    matches,
    # images,
)

keypoints_old = read_keypoints_hloc(features)

# from pixsfm.refine_hloc import PixSfM
# refiner = PixSfM()
# keypoints, _, _ = refiner.refine_keypoints(
#     path_to_output_keypoints.h5,
#     path_to_input_keypoints.h5,
#     path_to_list_of_image_pairs,
#     path_to_matches.h5,
#     path_to_image_dir,
# )
# print(keypoints_old)