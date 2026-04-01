#-----------------------------------------------------------------------------
set(MODULE_NAME AutomatedLandmarking)

#-----------------------------------------------------------------------------
# Only the main module script; mesh_conversion and fiducial_creation are helpers
# imported by AutomatedLandmarking.py, not standalone Slicer modules
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/requirements.txt
  Helpers/__init__.py
  Helpers/mesh_conversion.py
  Helpers/fiducial_creation.py
  )

# Bundled mvcnn package (BioFace3D Module 2) - Python code + model configs
file(GLOB_RECURSE MVCNN_RESOURCE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/Resources/mvcnn/*"
)
list(APPEND MODULE_PYTHON_RESOURCES ${MVCNN_RESOURCE_FILES})

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
