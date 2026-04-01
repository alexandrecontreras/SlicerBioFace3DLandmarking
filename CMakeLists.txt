cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(BioFace3D_Landmarking)

#-----------------------------------------------------------------------------
# Extension meta-information (keep in sync with README.md and Extensions Index JSON)
set(EXTENSION_HOMEPAGE "https://github.com/alexandrecontreras/SlicerBioFace3DLandmarking#readme")
set(EXTENSION_CONTRIBUTORS "Alexandre Contreras Urbita (La Salle - Universitat Ramon Llull, Spain)")
set(EXTENSION_DESCRIPTION "BioFace3D for 3D Slicer: automatic facial landmark detection on 3D facial meshes using BioFace3D MVCNN models, plus optional face mesh standardization to align surfaces to the reference space used for model inference. Standardization uses manually placed correspondences, a similarity transform, and optional ICP refinement. Landmark count depends on the selected model. Local PyTorch inference with optional CUDA acceleration; model weights are downloaded on demand if not already cached. MIT License (see LICENSE).")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/alexandrecontreras/SlicerBioFace3DLandmarking/main/icon.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/alexandrecontreras/SlicerBioFace3DLandmarking/main/Screenshots/standardization_before.png https://raw.githubusercontent.com/alexandrecontreras/SlicerBioFace3DLandmarking/main/Screenshots/standardization_after.png https://raw.githubusercontent.com/alexandrecontreras/SlicerBioFace3DLandmarking/main/Screenshots/landmarking_after.png")
set(EXTENSION_DEPENDS "NA") 

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(AutomatedLandmarking)
add_subdirectory(FaceStandardization)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
