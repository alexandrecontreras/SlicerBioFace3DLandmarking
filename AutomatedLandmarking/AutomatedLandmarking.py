import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import vtk

import ctk
import qt
import slicer

# Helpers subpackage
from Helpers import mesh_conversion, fiducial_creation, landmarks_io
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# Ensure Resources/ is on sys.path so "from mvcnn.api import ..." works
_MODULE_DIR = Path(__file__).resolve().parent
_BUNDLED_RESOURCES = _MODULE_DIR / "Resources"
if _BUNDLED_RESOURCES.is_dir() and str(_BUNDLED_RESOURCES) not in sys.path:
    sys.path.insert(0, str(_BUNDLED_RESOURCES))

# Pip specs. GPU torch: uses --index-url for cu124 
_TORCH_PIP_SPEC = "torch>=2.6.0,<2.11"
_SCIPY_PIP_SPEC = "scipy>=1.7.0"
_PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu124"


def _nvidia_smi_gpu_name():
    """Return first GPU name from nvidia-smi, or None if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        kwargs = {
            "args": [exe, "--query-gpu=name", "--format=csv,noheader"],
            "capture_output": True,
            "timeout": 8,
            "text": True,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(**kwargs)
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        return proc.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        return None


def torch_backend_status_message():
    """
    Human-readable PyTorch device line, whether CUDA is usable, and nvidia-smi GPU name if any.

    Returns (message, cuda_available, nvidia_smi_gpu_name).
    Third element is None if torch sees CUDA, or if nvidia-smi is missing / fails; used to gate the CUDA install button.
    """
    try:
        import torch
    except Exception as exc:
        nvidia_name = _nvidia_smi_gpu_name()
        return ("PyTorch is not available ({}).".format(exc), False, nvidia_name)

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA device"
        ver = getattr(torch.version, "cuda", None) or "?"
        return ("Using GPU: {} (CUDA {}).".format(name, ver), True, None)

    lines = ["PyTorch is using CPU only (slower)."]
    gpu_name = _nvidia_smi_gpu_name()
    if gpu_name:
        lines.append('NVIDIA driver reports "{}". Install GPU-enabled PyTorch below if landmarking is slow.'.format(gpu_name))
    else:
        lines.append("No NVIDIA GPU detected via nvidia-smi. GPU install is only for supported NVIDIA hardware.")
    return ("\n".join(lines), False, gpu_name)

'''=================================================================================================================='''
'''=================================================================================================================='''
#
# AutomatedLandmarking
#
class AutomatedLandmarking(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Landmark generation"
        self.parent.categories = ["BioFace3D"]
        self.parent.dependencies = []
        self.parent.contributors = ["Alex Contreras Urbita"]
        self.parent.helpText = (
            "Landmark generation runs automatic facial landmark detection on a 3D mesh using BioFace3D (Module 2). "
            "Select an input mesh (any format Slicer supports: PLY, OBJ, STL, etc.) and an output fiducial node, "
            "then click Generate Landmarks. Model configs ship with the extension, and model weights can be downloaded explicitly from the module when needed. "
            "Downloaded weights are cached locally on this computer and reused on later runs.\n\n"
            "By default PyTorch is installed as CPU-only (smaller download). For much faster inference on an NVIDIA GPU, "
            "open the Advanced tab and use Install GPU-enabled PyTorch (large download; restart Slicer afterward)."
        )
        self.parent.acknowledgementText = 'BioFace3D Slicer integration. \nPart of the BioFace3D_Landmarking extension.'

        print("AutomatedLandmarking(ScriptedLoadableModule):    __init__(self, parent)")

'''=================================================================================================================='''
'''=================================================================================================================='''
#
# AutomatedLandmarkingWidget
#
class AutomatedLandmarkingWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        """    Called when the user opens the module the first time and the widget is initialized.    """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None  # SingleTon initialized through self.setParameterNode(self.logic.getParameterNode())
        self._updatingGUIFromParameterNode = False
        self._observedFiducial = None  # Local reference to the output fiducial being observed
        print("**Widget.__init__(self, parent)")

    # ------------------------------------------------------------------------------------------------------------------
    def setup(self):
        print("**Widget.setup(self), \tBioFace3D_Landmarking")

        """    00. Called when the user opens the module the first time and the widget is initialized. """
        ScriptedLoadableModuleWidget.setup(self)

        # 00. Install BioFace3D dependencies if missing via pip (torch, scipy).
        needInstall = False
        try:
            import torch  
            import scipy 
        except ModuleNotFoundError:
            needInstall = True
        if needInstall:
            progressDialog = slicer.util.createProgressDialog(
                windowTitle="Installing...",
                labelText="Installing BioFace3D dependencies (torch, scipy). This may take some time...",
                maximum=0,
            )
            slicer.app.processEvents()
            try:
                slicer.util.pip_install([_TORCH_PIP_SPEC, _SCIPY_PIP_SPEC])
            except Exception as e:
                slicer.util.infoDisplay(
                    "Could not install BioFace3D dependencies. Please install manually:\n"
                    "  PythonSlicer -m pip install '{}' '{}'\n\n{}".format(_TORCH_PIP_SPEC, _SCIPY_PIP_SPEC, e)
                )
            progressDialog.close()

        # 01. Load widget from .ui file.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/AutomatedLandmarking.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # 02. Set scene in MRML widgets.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # 03. Create logic class.
        self.logic = AutomatedLandmarkingLogic()

        # 04. Connections, ensure parameter node is updated when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # 05a. Model management
        self.ui.modelSelector.currentIndexChanged.connect(self._refreshModelAvailabilityUI)
        self.ui.modelSelector.currentIndexChanged.connect(self._updateGenerateButtonState)
        self.ui.modelSelector.currentIndexChanged.connect(self._updateRunBatchButtonState)
        self.ui.downloadSelectedModelButton.clicked.connect(self.onDownloadSelectedModelButtonClicked)
        self.ui.refreshModelStatusButton.clicked.connect(self._refreshModelAvailabilityUI)
        self.ui.removeCachedModelButton.clicked.connect(self.onRemoveCachedModelButtonClicked)
        self.ui.statusLabel.text = "Select input mesh and output landmarks node, then click Generate Landmarks."

        # 05b. Single tab
        self.ui.inputModelSelector.currentNodeChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.outputFiducialSelector.currentNodeChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.generateLandmarksButton.clicked.connect(self.onGenerateLandmarksButton_Clicked)
        self.ui.thisMeshOnlyRadio.toggled.connect(self._updateGenerateButtonState)
        self.ui.allMeshesRadio.toggled.connect(self._updateGenerateButtonState)

        # 05c. Batch tab 
        self.ui.runBatchButton.clicked.connect(self.onRunBatchButton_Clicked)
        self.ui.batchInputDirEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.ui.batchOutputDirEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.ui.batchInputDirEdit.connect("currentPathChanged(QString)", self._updateRunBatchButtonState)
        self.ui.batchOutputDirEdit.connect("currentPathChanged(QString)", self._updateRunBatchButtonState)
        self._updateRunBatchButtonState()

        self.ui.installGpuTorchButton.clicked.connect(self.onInstallGpuTorchButtonClicked)
        self._updateTorchBackendGroup()

        # 05c. Populate model selector from __configs 
        configs_dir = _BUNDLED_RESOURCES / "mvcnn" / "__configs"
        default_model = "21Landmarks_25views"
        if configs_dir.is_dir():
            for d in sorted(configs_dir.iterdir()):
                if d.is_dir() and (d / "config.json").is_file():
                    self.ui.modelSelector.addItem(d.name, d.name)
        if self.ui.modelSelector.count == 0:
            self.ui.modelSelector.addItem(default_model, default_model)
        # Set default to 21Landmarks_25views
        idx = self.ui.modelSelector.findText(default_model)
        if idx >= 0:
            self.ui.modelSelector.setCurrentIndex(idx)
        self._refreshModelAvailabilityUI()

        # 06. Needed for programmer-friendly  Module-Reload
        if self.parent.isEntered:
            self.initializeParameterNode()

    # ------------------------------------------------------------------------------------------------------------------
    def cleanup(self):
        """    Called when the application closes and the module widget is destroyed.    """
        print("**Widget.cleanup(self)")
        self.removeObservers()

    # ------------------------------------------------------------------------------------------------------------------
    def _updateTorchBackendGroup(self):
        """Refresh PyTorch status label and GPU install button (Advanced tab)."""
        if not getattr(self, "ui", None):
            return
        msg, cuda_ok, nvidia_name = torch_backend_status_message()
        self.ui.torchBackendStatusLabel.text = msg
        is_macos = sys.platform == "darwin"
        can_offer_cuda_wheels = (not cuda_ok) and (nvidia_name is not None) and (not is_macos)
        self.ui.installGpuTorchButton.enabled = can_offer_cuda_wheels
        if cuda_ok:
            tip = "PyTorch is already using CUDA."
        elif is_macos:
            tip = (
                "NVIDIA CUDA PyTorch wheels apply to Windows and Linux. "
                "On macOS this module uses CPU PyTorch (Apple GPUs use a different stack, not this installer)."
            )
        elif nvidia_name is None:
            tip = (
                "No NVIDIA GPU was detected (nvidia-smi). "
                "This action installs PyTorch with NVIDIA CUDA; it is only offered when an NVIDIA driver reports a GPU."
            )
        else:
            tip = (
                "Reinstall PyTorch from the CUDA wheel index (large download, often 1 to 3 GB). "
                "Requires a recent NVIDIA driver. Restart 3D Slicer when finished.\n\n"
                "Index: {}".format(_PYTORCH_CUDA_INDEX_URL)
            )
        self.ui.installGpuTorchButton.toolTip = tip

    # ------------------------------------------------------------------------------------------------------------------
    def onInstallGpuTorchButtonClicked(self):
        if not getattr(self, "ui", None):
            return
        mw = slicer.util.mainWindow()
        prompt = (
            "This will reinstall PyTorch from the official CUDA wheel index (replaces CPU-only +cpu builds). "
            "The download is large (often 1 to 3 GB) and requires a recent NVIDIA driver.\n\n"
            "SciPy is left unchanged. Restart 3D Slicer after installation.\n\n"
            "Continue?"
        )
        if qt.QMessageBox.question(mw, "Install GPU-enabled PyTorch", prompt, qt.QMessageBox.Yes | qt.QMessageBox.No, qt.QMessageBox.No) != qt.QMessageBox.Yes:
            return
        progress = slicer.util.createProgressDialog(
            windowTitle="Installing...",
            labelText="Installing GPU-enabled PyTorch (this may take several minutes)...",
            maximum=0,
        )
        slicer.app.processEvents()
        manual = (
            "PythonSlicer -m pip install --upgrade --force-reinstall --no-cache-dir --index-url {} '{}'".format(
                _PYTORCH_CUDA_INDEX_URL, _TORCH_PIP_SPEC
            )
        )
        try:
            # --index-url: resolve torch from CUDA index only.
            # --no-cache-dir: avoid reinstalling a previously cached CPU wheel. --force-reinstall: replace +cpu install.
            slicer.util.pip_install(
                [
                    "--upgrade",
                    "--force-reinstall",
                    "--no-cache-dir",
                    "--index-url",
                    _PYTORCH_CUDA_INDEX_URL,
                    _TORCH_PIP_SPEC,
                ]
            )
        except Exception as e:
            progress.close()
            slicer.util.errorDisplay(
                "GPU PyTorch install failed. Try from a terminal:\n\n{}\n\n{}".format(manual, e)
            )
            self._updateTorchBackendGroup()
            return
        progress.close()
        slicer.util.infoDisplay(
            "GPU-enabled PyTorch was installed.\n\n"
            "Quit and restart 3D Slicer, then open this module again to confirm "
            'the status shows "Using GPU".'
        )
        self._updateTorchBackendGroup()

    # ------------------------------------------------------------------------------------------------------------------
    def enter(self):
        """    Called each time the user opens this module.    """
        print("\n**Widget.enter(self)")
        self.initializeParameterNode()
        self._updateTorchBackendGroup()
        self._refreshModelAvailabilityUI()

    # ------------------------------------------------------------------------------------------------------------------
    def exit(self):
        """    Called each time the user opens a different module.    """
        print("**Widget.exit(self)")
        if self._parameterNode:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        if self._observedFiducial:
            self.removeObserver(self._observedFiducial, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onOutputFiducialModified)
            self.removeObserver(self._observedFiducial, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onOutputFiducialModified)
            self.removeObserver(self._observedFiducial, slicer.vtkMRMLMarkupsNode.PointRemovedEvent, self.onOutputFiducialModified)

    # ------------------------------------------------------------------------------------------------------------------
    def onSceneStartClose(self, caller, event):
        """    Called just before the scene is closed.    """
        print("**Widget.onSceneStartClose(self, caller, event)")
        self.setParameterNode(None)

    # ------------------------------------------------------------------------------------------------------------------
    def onSceneEndClose(self, caller, event):
        """     Called just after the scene is closed.    """
        print("**Widget.onSceneEndClose(self, caller, event)")
        if self.parent.isEntered:
            self.initializeParameterNode()

    # ------------------------------------------------------------------------------------------------------------------
    def initializeParameterNode(self):
        """    Ensure parameter node exists and observed. """
        print("\t**Widget.initializeParameterNode(self), \t BioFace3D_Landmarking")
        self.setParameterNode(self.logic.getParameterNode())

    # ------------------------------------------------------------------------------------------------------------------
    def setParameterNode(self, inputParameterNode):
        """    Set and observe the SingleTon ParameterNode. """
        print("\t\t**Widget.setParameterNode(self, inputParameterNode)")
        if inputParameterNode:
            if not inputParameterNode.IsSingleton():
                raise ValueError('BioFace3D_Landmarking Alert! \tinputParameterNode is not a singleton!')
            self.logic.setDefaultParameters(inputParameterNode)

        # 01. Unobserve previously selected SingleTon ParameterNode
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # 02. Set new SingleTon ParameterNode and add observer
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # 03. Initial GUI update
        self.updateGUIFromParameterNode()

    # ------------------------------------------------------------------------------------------------------------------
    def updateGUIFromParameterNode(self, caller=None, event=None):
        """   Update GUI from ParameterNode. Includes node observation management. """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # I. Open-Brace: Prevent infinite loops
        self._updatingGUIFromParameterNode = True

        print("**Widget.updateGUIFromParameterNode(self, caller=None, event=None), \tBioFace3D_Landmarking")

        # II. Sync GUI from parameter node refs
        inputMesh = self._parameterNode.GetNodeReference("InputMesh")
        outputFiducial = self._parameterNode.GetNodeReference("OutputFiducial")
        self.ui.inputModelSelector.setCurrentNode(inputMesh)
        self.ui.outputFiducialSelector.setCurrentNode(outputFiducial)

        # III. Handle output fiducial observation (for status label)
        if outputFiducial != self._observedFiducial:
            if self._observedFiducial:
                self.removeObserver(self._observedFiducial, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onOutputFiducialModified)
                self.removeObserver(self._observedFiducial, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onOutputFiducialModified)
                self.removeObserver(self._observedFiducial, slicer.vtkMRMLMarkupsNode.PointRemovedEvent, self.onOutputFiducialModified)
            self._observedFiducial = outputFiducial
            if self._observedFiducial:
                self.addObserver(self._observedFiducial, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onOutputFiducialModified)
                self.addObserver(self._observedFiducial, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self.onOutputFiducialModified)
                self.addObserver(self._observedFiducial, slicer.vtkMRMLMarkupsNode.PointRemovedEvent, self.onOutputFiducialModified)

        # IV. Trigger status update and button state
        self.onOutputFiducialModified()
        self._updateGenerateButtonState()

        # V. Close-Brace
        self._updatingGUIFromParameterNode = False

    # ------------------------------------------------------------------------------------------------------------------
    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """ Save GUI selection into ParameterNode. """
        print("**Widget.updateParameterNodeFromGUI(self, caller=None, event=None),     \t BioFace3D_Landmarking")
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        wasModified = self._parameterNode.StartModify()
        self._parameterNode.SetNodeReferenceID("InputMesh", self.ui.inputModelSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputFiducial", self.ui.outputFiducialSelector.currentNodeID)
        self._parameterNode.EndModify(wasModified)
        self._updateGenerateButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    def _updateRunBatchButtonState(self, *_args):
        """ Enable Run batch only when input and output directories are set; hint / ready text in status. """
        input_dir = (self.ui.batchInputDirEdit.currentPath or "").strip()
        output_dir = (self.ui.batchOutputDirEdit.currentPath or "").strip()
        ok = bool(input_dir and output_dir)
        model_status = self._get_selected_model_status()
        if not ok:
            self.ui.runBatchButton.enabled = False
            self.ui.batchStatusLabel.text = "Select input and output directories, then click Run batch."
        elif model_status and not model_status["is_available"] and model_status["download_url"]:
            self.ui.runBatchButton.enabled = False
            self.ui.batchStatusLabel.text = "Download the selected model above to enable batch prediction."
        elif model_status and not model_status["is_available"]:
            self.ui.runBatchButton.enabled = False
            self.ui.batchStatusLabel.text = "Selected model is not available locally."
        else:
            self.ui.runBatchButton.enabled = True
            self.ui.batchStatusLabel.text = "Click Run batch to process all meshes in the input folder."

    # ------------------------------------------------------------------------------------------------------------------
    def _updateGenerateButtonState(self):
        """ Enable Generate Landmarks when valid: single mesh compatible, or (all meshes) at least one mesh in scene. """
        if self._parameterNode is None or self.logic is None:
            self.ui.generateLandmarksButton.enabled = False
            return
        model_status = self._get_selected_model_status()
        if model_status and not model_status["is_available"] and model_status["download_url"]:
            self.ui.generateLandmarksButton.toolTip = (
                "Download the selected model to enable prediction."
            )
        elif model_status and not model_status["is_available"]:
            self.ui.generateLandmarksButton.toolTip = "The selected model is not available locally."
        else:
            self.ui.generateLandmarksButton.toolTip = "Generate landmarks using the selected BioFace3D model."
        if model_status and not model_status["is_available"]:
            self.ui.generateLandmarksButton.enabled = False
            if model_status["download_url"]:
                self.ui.statusLabel.text = "Download the selected model to enable prediction."
            else:
                self.ui.statusLabel.text = "Selected model is not available locally."
            return
        allMeshes = self.ui.allMeshesRadio.isChecked()
        if allMeshes:
            n = 0
            for i in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLModelNode")):
                node = slicer.mrmlScene.GetNthNodeByClass(i, "vtkMRMLModelNode")
                name = node.GetName() or ""
                if "Volume Slice" in name:
                    continue
                poly = node.GetPolyData()
                if poly and poly.GetNumberOfPoints() > 0:
                    n += 1
            self.ui.generateLandmarksButton.enabled = n > 0
            if n == 0:
                self.ui.statusLabel.text = "No meshes in scene. Load meshes to process."
        else:
            inputMesh = self._parameterNode.GetNodeReference("InputMesh")
            compatible, msg = self.logic.isMeshCompatibleForLandmarking(inputMesh)
            self.ui.generateLandmarksButton.enabled = compatible
            if not compatible and msg:
                self.ui.statusLabel.text = msg

    # ------------------------------------------------------------------------------------------------------------------
    def onOutputFiducialModified(self, caller=None, event=None):
        """ Update status label from output fiducial. """
        print("\t\t**Widget.onOutputFiducialModified(self)")
        if not self._observedFiducial:
            model_status = self._get_selected_model_status()
            if model_status and not model_status["is_available"]:
                if model_status["download_url"]:
                    self.ui.statusLabel.text = "Download the selected model to enable prediction."
                else:
                    self.ui.statusLabel.text = "Selected model is not available locally."
                return
            if self._parameterNode and self._parameterNode.GetNodeReference("InputMesh"):
                compatible, _ = self.logic.isMeshCompatibleForLandmarking(self._parameterNode.GetNodeReference("InputMesh"))
                if compatible:
                    self.ui.statusLabel.text = "Select output landmarks node and click Generate Landmarks."
                else:
                    self.ui.statusLabel.text = "Selected mesh has no geometry. Load or choose a valid mesh."
            else:
                self.ui.statusLabel.text = "Select input mesh and output landmarks node, then click Generate Landmarks."
            return
        n = self._observedFiducial.GetNumberOfControlPoints()
        self.ui.statusLabel.text = "Landmarks: {} | Points: {}".format(self._observedFiducial.GetName(), n)

    # ------------------------------------------------------------------------------------------------------------------
    def _get_model_dir_from_ui(self):
        """ Resolve model dir from model selector. Returns Path or None. """
        model_name = self.ui.modelSelector.currentText
        if not model_name:
            return None
        path = _BUNDLED_RESOURCES / "mvcnn" / "__configs" / model_name
        return path if path.is_dir() else None

    def _get_max_ransac_from_ui(self):
        """ Max Ransac Error from Advanced tab. """
        return self.ui.maxRansacSpinBox.value

    def _get_predict_num_from_ui(self):
        """ Mean Predictions from Advanced tab. """
        return self.ui.meanPredictionsSpinBox.value

    def _get_predict_tries_from_ui(self):
        """ Max Tries from Advanced tab. """
        return self.ui.maxTriesSpinBox.value

    def _get_selected_model_status(self):
        """Return status dict for the currently selected model, or None on error."""
        model_dir = self._get_model_dir_from_ui()
        if not model_dir or not model_dir.is_dir():
            return None
        try:
            from mvcnn.api import get_model_status
            return get_model_status(model_dir)
        except Exception as exc:
            logging.warning("AutomatedLandmarking: could not inspect model status for %s: %s", model_dir, exc)
            return None

    def _refreshModelAvailabilityUI(self, *_args):
        """Refresh inline model availability labels and action buttons."""
        if not getattr(self, "ui", None):
            return
        status = self._get_selected_model_status()
        if status is None:
            self.ui.modelStatusLabel.text = "Status: Unavailable"
            self.ui.downloadSelectedModelButton.enabled = False
            self.ui.refreshModelStatusButton.enabled = False
            self.ui.removeCachedModelButton.enabled = False
            self.ui.modelSelector.toolTip = "Choose a valid bundled model configuration."
            self.ui.modelStatusLabel.toolTip = self.ui.modelSelector.toolTip
            self._updateGenerateButtonState()
            self._updateRunBatchButtonState()
            return

        display_name = status["display_name"]
        if status["recommended"]:
            display_name += " (recommended)"

        wh = status.get("weights_size_human")
        dh = status.get("download_size_human")

        if status["is_available"]:
            source_text = {
                "bundled": "bundled with the extension",
                "cached": "cached locally",
                "local": "available from a local path",
            }.get(status["availability_source"], "available locally")
            line = "Status: Ready ({})".format(source_text)
            if wh:
                line += " — Weights: {}".format(wh)
            self.ui.modelStatusLabel.text = line
        elif status.get("validation_error"):
            line = "Status: Cached model is invalid"
            if wh:
                line += " — Expected file: {}".format(wh)
            self.ui.modelStatusLabel.text = line
        elif status["download_url"]:
            line = "Status: Not downloaded"
            if dh:
                line += " — Approx. download: {}".format(dh)
            self.ui.modelStatusLabel.text = line
        else:
            self.ui.modelStatusLabel.text = "Status: Unavailable locally"

        detail_lines = ["Selected model: {}".format(display_name)]
        if status["description"]:
            detail_lines.append(status["description"])
        if status["landmark_count"]:
            detail_lines.append("Landmarks: {}".format(status["landmark_count"]))
        if status["is_available"] and wh:
            detail_lines.append("Weights file: {}".format(wh))
        elif dh and not status["is_available"]:
            detail_lines.append("Approx. download size: {}".format(dh))
        elif wh:
            detail_lines.append("Weights file: {}".format(wh))
        if status.get("validation_error"):
            detail_lines.append("Issue: {}".format(status["validation_error"]))
        self.ui.modelSelector.toolTip = "\n".join(detail_lines)
        self.ui.modelStatusLabel.toolTip = self.ui.modelSelector.toolTip

        self.ui.downloadSelectedModelButton.enabled = (not status["is_available"]) and bool(status["download_url"])
        self.ui.refreshModelStatusButton.enabled = True
        self.ui.removeCachedModelButton.enabled = status["availability_source"] == "cached"
        self._updateGenerateButtonState()
        self._updateRunBatchButtonState()

    def _download_model_for_status(self, status, show_success_message=True):
        """Download the selected model with a visible progress dialog."""
        if not status:
            slicer.util.warningDisplay("Select a valid model first.")
            return False
        if status["is_available"]:
            if show_success_message:
                slicer.util.infoDisplay("The selected model is already available locally.")
            self._refreshModelAvailabilityUI()
            return True
        if not status["download_url"]:
            slicer.util.errorDisplay("This model has no download URL configured.")
            return False

        progress = slicer.util.createProgressDialog(
            windowTitle="Downloading model...",
            labelText=(
                'Downloading "{}". This may take a while depending on the model size and your connection...'.format(
                    status["display_name"]
                )
            ),
            maximum=0,
        )
        progress.setMinimumDuration(0)
        slicer.app.processEvents()

        try:
            from mvcnn.api import download_model
            download_model(status["model_dir"])
        except Exception as exc:
            progress.close()
            self._refreshModelAvailabilityUI()
            slicer.util.errorDisplay("Model download failed:\n\n{}".format(exc))
            return False

        progress.close()
        self._refreshModelAvailabilityUI()
        if show_success_message:
            slicer.util.infoDisplay(
                'Model "{}" is ready and will be reused from local cache on future runs.'.format(status["display_name"])
            )
        return True

    def onDownloadSelectedModelButtonClicked(self):
        """Explicitly download the selected model before running inference."""
        status = self._get_selected_model_status()
        if status and status["is_available"]:
            slicer.util.infoDisplay("The selected model is already ready to use.")
            self._refreshModelAvailabilityUI()
            return
        if not status:
            slicer.util.warningDisplay("Select a valid model first.")
            return
        prompt = (
            'Download model "{}" now?\n\n'
            "The weights will be stored in your local BioFace3D cache and reused for future predictions.".format(
                status["display_name"]
            )
        )
        if qt.QMessageBox.question(
            slicer.util.mainWindow(),
            "Download model",
            prompt,
            qt.QMessageBox.Yes | qt.QMessageBox.No,
            qt.QMessageBox.Yes,
        ) != qt.QMessageBox.Yes:
            return
        self._download_model_for_status(status, show_success_message=True)

    def onRemoveCachedModelButtonClicked(self):
        """Remove the selected model from the local cache."""
        status = self._get_selected_model_status()
        if not status:
            slicer.util.warningDisplay("Select a valid model first.")
            return
        if status["availability_source"] != "cached":
            slicer.util.infoDisplay("This model is not currently cached locally.")
            self._refreshModelAvailabilityUI()
            return
        prompt = (
            'Remove cached weights for "{}"?\n\n'
            "This only deletes the local cache copy. You can download it again later.".format(status["display_name"])
        )
        if qt.QMessageBox.question(
            slicer.util.mainWindow(),
            "Remove cached model",
            prompt,
            qt.QMessageBox.Yes | qt.QMessageBox.No,
            qt.QMessageBox.No,
        ) != qt.QMessageBox.Yes:
            return
        try:
            from mvcnn.api import remove_cached_model
            removed = remove_cached_model(status["model_dir"])
        except Exception as exc:
            slicer.util.errorDisplay("Could not remove cached model:\n\n{}".format(exc))
            return
        self._refreshModelAvailabilityUI()
        if removed:
            slicer.util.infoDisplay("Cached model removed.")
        else:
            slicer.util.infoDisplay("No cached weights were found for the selected model.")

    # ------------------------------------------------------------------------------------------------------------------
    def onGenerateLandmarksButton_Clicked(self):
        """ Run landmark prediction and fill output fiducial(s). """
        print("**Widget.onGenerateLandmarksButton_Clicked(self)")
        status = self._get_selected_model_status()
        if not status or not status["is_available"]:
            slicer.util.warningDisplay("Download the selected model before running prediction.")
            return
        self.ui.generateLandmarksButton.enabled = False
        self.ui.statusLabel.text = "Running landmark prediction..."
        progressWidget = self.ui.progressWidget
        progressBar = self.ui.progressBar

        def progressCallback(current, total, message):
            progressBar.setMaximum(total)
            progressBar.setValue(current)
            progressBar.setFormat("%s (%d/%d)" % (message, current, total))
            slicer.app.processEvents()

        progressWidget.setVisible(True)
        progressBar.setValue(0)
        try:
            allMeshes = self.ui.allMeshesRadio.isChecked()
            if allMeshes:
                meshList = []
                for i in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLModelNode")):
                    node = slicer.mrmlScene.GetNthNodeByClass(i, "vtkMRMLModelNode")
                    name = node.GetName() or ""
                    if "Volume Slice" in name:
                        continue
                    poly = node.GetPolyData()
                    if poly and poly.GetNumberOfPoints() > 0:
                        meshList.append(node)
            else:
                meshList = None
            model_dir = self._get_model_dir_from_ui()
            max_ransac = self._get_max_ransac_from_ui()
            predict_num = self._get_predict_num_from_ui()
            predict_tries = self._get_predict_tries_from_ui()
            with slicer.util.tryWithErrorDisplay("Failed to run landmark prediction.", waitCursor=True):
                outputNodes = self.logic.runLandmarkPrediction(
                    meshList=meshList,
                    progressCallback=progressCallback,
                    model_dir=model_dir,
                    max_ransac_error=max_ransac,
                    predict_num=predict_num,
                    predict_tries=predict_tries,
                )
                if outputNodes:
                    outputNode = outputNodes[0] if isinstance(outputNodes, (list, tuple)) else outputNodes
                    wasModified = self._parameterNode.StartModify()
                    self._parameterNode.SetNodeReferenceID("OutputFiducial", outputNode.GetID())
                    self._parameterNode.EndModify(wasModified)
                    n = len(outputNodes) if isinstance(outputNodes, (list, tuple)) else 1
                    pts = outputNode.GetNumberOfControlPoints()
                    self.ui.statusLabel.text = "Generated landmarks for {} mesh(es). Points: {}.".format(n, pts)
                else:
                    self.ui.statusLabel.text = "Prediction failed or no landmarks returned."
        finally:
            progressWidget.setVisible(False)
            progressBar.setValue(progressBar.maximum)
            self._updateGenerateButtonState()

    # ------------------------------------------------------------------------------------------------------------------
    def onRunBatchButton_Clicked(self):
        """ Run batch landmark prediction on mesh files in input directory. """
        input_dir = self.ui.batchInputDirEdit.currentPath
        output_dir = self.ui.batchOutputDirEdit.currentPath
        if not input_dir or not output_dir:
            self.ui.batchStatusLabel.text = "Select input and output directories."
            slicer.util.warningDisplay("Select input and output directories.")
            return

        output_format = self.ui.batchOutputFormatCombo.currentText or "txt"
        model_dir = self._get_model_dir_from_ui()
        status = self._get_selected_model_status()
        if not status or not status["is_available"]:
            slicer.util.warningDisplay("Download the selected model before running batch prediction.")
            return

        progressWidget = self.ui.batchProgressWidget
        progressBar = self.ui.batchProgressBar
        batchStatusLabel = self.ui.batchStatusLabel

        def progressCallback(current, total, message):
            progressBar.setMaximum(total)
            progressBar.setValue(current)
            progressBar.setFormat("%s (%d/%d)" % (message, current, total))
            slicer.app.processEvents()

        self.ui.runBatchButton.enabled = False
        batchStatusLabel.text = "Running batch..."
        progressWidget.setVisible(True)
        progressBar.setValue(0)
        batch_summary = None
        try:
            with slicer.util.tryWithErrorDisplay("Batch processing failed.", waitCursor=True):
                success, fail = self.logic.runLandmarkPredictionBatch(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    model_dir=model_dir,
                    output_format=output_format,
                    progressCallback=progressCallback,
                )
                batch_summary = "Batch complete: %d succeeded, %d failed." % (success, fail)
                batchStatusLabel.text = batch_summary
        finally:
            progressWidget.setVisible(False)
            progressBar.setValue(progressBar.maximum)
            self._updateRunBatchButtonState()
            if batch_summary:
                batchStatusLabel.text = batch_summary

'''=================================================================================================================='''
'''=================================================================================================================='''
#
# AutomatedLandmarkingLogic
#
class AutomatedLandmarkingLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        print("**Logic.__init__(self)")

    # ------------------------------------------------------------------------------------------------------------------
    def setDefaultParameters(self, parameterNode):
        """    Initialize parameter node with defaults if empty.    """
        print("\t\t\t**Logic.setDefaultParameters(self, parameterNode), \tBioFace3D_Landmarking")
        pass

    # ------------------------------------------------------------------------------------------------------------------
    def isMeshCompatibleForLandmarking(self, mesh_node):
        """
        Return whether the model node can be used for landmarking.
        Compatible = vtkMRMLModelNode with poly data and at least one point.

        Returns tuple (bool, str): (True, None) if compatible; (False, message) if not.
        """
        if mesh_node is None:
            return False, "Select an input mesh."
        poly_data = mesh_node.GetPolyData()
        if poly_data is None:
            return False, "Selected node has no poly data. Choose a valid mesh."
        if poly_data.GetNumberOfPoints() == 0:
            return False, "Selected mesh has no points. Choose a valid mesh."
        return True, None

    # ------------------------------------------------------------------------------------------------------------------
    def runLandmarkPrediction(
        self,
        meshList=None,
        progressCallback=None,
        model_dir=None,
        max_ransac_error=5.0,
        predict_num=1,
        predict_tries=3,
    ):
        """
        Run BioFace3D landmark prediction on mesh(es).

        Parameters
        ----------
        meshList : list of vtkMRMLModelNode or None
            If None: use InputMesh from parameter node, return single fiducial node.
            If list: run on each mesh, create one fiducial per mesh, return list of fiducial nodes.
        progressCallback : callable or None
            Optional fn(current, total, message) for progress feedback.
        model_dir : pathlib.Path or None
            Path to model config dir (e.g. __configs/21Landmarks_25views). If None, uses default.
        max_ransac_error : float
            RANSAC error threshold. Default 5.0.
        predict_num : int
            Number of prediction runs to average. Default 1 (matches old module).
        predict_tries : int
            Max retries per run when ransac exceeds threshold. Default 3.

        Returns
        -------
        vtkMRMLMarkupsFiducialNode or list of vtkMRMLMarkupsFiducialNode or None
        """
        print("\t\t\t**Logic.runLandmarkPrediction(self, meshList=%s)" % (meshList,))
        parameterNode = self.getParameterNode()
        if not parameterNode:
            return None
        if meshList is None:
            inputMesh = parameterNode.GetNodeReference("InputMesh")
            if not inputMesh:
                logging.warning("AutomatedLandmarking: No input mesh selected.")
                return None
            compatible, msg = self.isMeshCompatibleForLandmarking(inputMesh)
            if not compatible:
                logging.warning("AutomatedLandmarking: %s", msg or "Mesh not compatible.")
                return None
            meshList = [inputMesh]
        elif len(meshList) == 0:
            return []

        outputFiducial = parameterNode.GetNodeReference("OutputFiducial") if len(meshList) == 1 else None

        # Reload Helpers so edits are picked up without restarting Slicer
        import importlib
        importlib.reload(mesh_conversion)

        outputNodes = []
        total = len(meshList)
        for idx, inputMesh in enumerate(meshList):
            if progressCallback:
                progressCallback(idx + 1, total, "Processing %s" % (inputMesh.GetName() or "mesh"))
            compatible, msg = self.isMeshCompatibleForLandmarking(inputMesh)
            if not compatible:
                logging.warning("AutomatedLandmarking: Skipping %s - %s", inputMesh.GetName(), msg or "not compatible")
                continue

            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                ply_path = tmp.name
            try:
                mesh_conversion.slicer_mesh_to_ply(inputMesh, ply_path)
            except (ValueError, IOError) as e:
                logging.warning("AutomatedLandmarking: export to PLY failed for %s: %s", inputMesh.GetName(), e)
                continue

            try:
                mdl_dir = model_dir or self._get_model_dir()
                landmarks = None
                if mdl_dir and mdl_dir.is_dir():
                    try:
                        from mvcnn.api import predict_landmarks
                        landmarks = predict_landmarks(
                            str(ply_path),
                            str(mdl_dir),
                            use_gpu=True,
                            predict_num=predict_num,
                            max_ransac_error=max_ransac_error,
                            predict_tries=predict_tries,
                        )
                    except (ImportError, ModuleNotFoundError) as e:
                        logging.warning("AutomatedLandmarking: missing dependency - %s", e)
                        if idx == 0:
                            slicer.util.infoDisplay(
                                "BioFace3D import failed (torch/scipy may be missing or another dependency failed):\n\n"
                                "{0}\n\n"
                                "If torch/scipy are missing, install with:\n"
                                "  PythonSlicer -m pip install '{1}' '{2}'\n\n"
                                "For NVIDIA GPU builds, use Advanced - Install GPU-enabled PyTorch, or:\n"
                                "  PythonSlicer -m pip install --upgrade --force-reinstall --no-cache-dir --index-url {3} '{1}'".format(
                                    e,
                                    _TORCH_PIP_SPEC,
                                    _SCIPY_PIP_SPEC,
                                    _PYTORCH_CUDA_INDEX_URL,
                                )
                            )
                    except Exception as e:
                        logging.warning("AutomatedLandmarking: predict_landmarks failed for %s: %s", inputMesh.GetName(), e)

                if landmarks is not None and len(landmarks) > 0:
                    fid = fiducial_creation.landmarks_to_fiducial(
                        landmarks,
                        fiducial_node=outputFiducial if len(meshList) == 1 and idx == 0 else None,
                        coordinate_system="LPS",
                    )
                    if len(meshList) > 1:
                        fid.SetName((inputMesh.GetName() or "mesh") + "_landmarks")
                    outputNodes.append(fid)
                else:
                    logging.warning(
                        "AutomatedLandmarking: Prediction failed for %s — skipped (no placeholder created)",
                        inputMesh.GetName() or "mesh",
                    )
            finally:
                if os.path.isfile(ply_path):
                    try:
                        os.remove(ply_path)
                    except OSError:
                        pass

        if len(meshList) == 1:
            return outputNodes[0] if outputNodes else None
        return outputNodes

    # ------------------------------------------------------------------------------------------------------------------
    def _get_model_dir(self):
        """ Resolve path to BioFace3D model config (21Landmarks_25views). Uses bundled Resources/mvcnn. """
        model_dir = _BUNDLED_RESOURCES / "mvcnn" / "__configs" / "21Landmarks_25views"
        if model_dir.is_dir():
            return model_dir
        return None

    # ------------------------------------------------------------------------------------------------------------------
    def runLandmarkPredictionBatch(
        self,
        input_dir,
        output_dir,
        model_dir=None,
        output_format="txt",
        progressCallback=None,
    ):
        """
        Process all mesh files in input_dir, run landmark prediction, save to output_dir.
        output_format: 'txt', 'fcsv', 'landmarkAscii', or 'all'.
        Returns (success_count, fail_count).
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        if not input_dir.is_dir():
            logging.warning("AutomatedLandmarking: Input directory not found: %s", input_dir)
            return 0, 0
        output_dir.mkdir(parents=True, exist_ok=True)

        mesh_extensions = {".ply", ".obj", ".stl", ".vtk", ".wrl"}
        mesh_files = [
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in mesh_extensions
        ]
        mesh_files = sorted(mesh_files)

        mdl_dir = model_dir or self._get_model_dir()
        if not mdl_dir or not mdl_dir.is_dir():
            logging.warning("AutomatedLandmarking: Model directory not found")
            return 0, len(mesh_files)

        success_count = 0
        fail_count = 0
        total = len(mesh_files)

        for idx, mesh_path in enumerate(mesh_files):
            if progressCallback:
                progressCallback(idx + 1, total, "Processing %s" % mesh_path.name)
            try:
                from mvcnn.api import predict_landmarks
                landmarks = predict_landmarks(
                    str(mesh_path),
                    str(mdl_dir),
                    use_gpu=True,
                )
            except Exception as e:
                logging.warning("AutomatedLandmarking: predict_landmarks failed for %s: %s", mesh_path.name, e)
                fail_count += 1
                continue

            if landmarks is None or len(landmarks) == 0:
                logging.warning("AutomatedLandmarking: Prediction failed for %s", mesh_path.name)
                fail_count += 1
                continue

            basename = mesh_path.stem
            try:
                if output_format == "txt":
                    landmarks_io.write_landmarks_txt(landmarks, output_dir / (basename + "_landmarks.txt"))
                elif output_format == "fcsv":
                    landmarks_io.write_landmarks_fcsv(landmarks, output_dir / (basename + "_landmarks.fcsv"))
                elif output_format == "landmarkAscii":
                    landmarks_io.write_landmarks_ascii(landmarks, output_dir / (basename + "_landmarks.landmarkAscii"))
                elif output_format == "all":
                    landmarks_io.write_landmarks_txt(landmarks, output_dir / (basename + "_landmarks.txt"))
                    landmarks_io.write_landmarks_fcsv(landmarks, output_dir / (basename + "_landmarks.fcsv"))
                    landmarks_io.write_landmarks_ascii(landmarks, output_dir / (basename + "_landmarks.landmarkAscii"))
                else:
                    landmarks_io.write_landmarks_txt(landmarks, output_dir / (basename + "_landmarks.txt"))
                success_count += 1
            except Exception as e:
                logging.warning("AutomatedLandmarking: Failed to save landmarks for %s: %s", mesh_path.name, e)
                fail_count += 1

        return success_count, fail_count

'''=================================================================================================================='''
'''=================================================================================================================='''
#
# AutomatedLandmarkingTest
#
class AutomatedLandmarkingTest(ScriptedLoadableModuleTest):

    def setUp(self):
        slicer.mrmlScene.Clear()

    # ------------------------------------------------------------------------------------------------------------------
    def runTest(self):
        self.setUp()
        self.test_AutomatedLandmarking_LogicExists()

    # ------------------------------------------------------------------------------------------------------------------
    def test_AutomatedLandmarking_LogicExists(self):
        self.delayDisplay("Starting the test")
        logic = AutomatedLandmarkingLogic()
        parameterNode = logic.getParameterNode()
        self.assertIsNotNone(parameterNode)
        self.delayDisplay('Test passed')
