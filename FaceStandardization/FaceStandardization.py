import logging
from pathlib import Path

import vtk

import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# Fixed names for bundled reference - check with GetFirstNodeByName before loading
REF_MESH_NAME = "FaceStandardization_reference"
REF_LANDMARKS_NAME = "FaceStandardization_reference_landmarks"

_MODULE_DIR = Path(__file__).resolve().parent
_RESOURCES_DIR = _MODULE_DIR / "Resources"

'''=================================================================================================================='''
'''=================================================================================================================='''
#
# FaceStandardization
#
class FaceStandardization(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Face Mesh Standardization"
        self.parent.categories = ["BioFace3D"]
        self.parent.dependencies = []
        self.parent.contributors = ["Alex Contreras Urbita"]
        self.parent.helpText = """
        Standardize facial meshes for AI landmark detection.

        Single tab (scene-based):
        1. Select source mesh (the mesh you place landmarks on).
        2. Create Source Landmarks, place exactly 5 points (same order as reference 1-5).
        3. Choose Apply to: This mesh only, or All meshes in scene.
        4. Click Apply Standardization.

        Batch tab (file-based): Input dir, reference landmarks .fcsv, output dir. Run batch.
        """
        self.parent.acknowledgementText = "Uses vtkLandmarkTransform + vtkIterativeClosestPointTransform (optionally)."

        print("FaceStandardization(ScriptedLoadableModule):    __init__(self, parent)")

'''=================================================================================================================='''
'''=================================================================================================================='''
#
# FaceStandardizationWidget
#
class FaceStandardizationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        """    Called when the user opens the module the first time and the widget is initialized.    """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._observedSourceLm = None
        self._refMesh = None
        self._refLandmarks = None
        print("**Widget.__init__(self, parent)")

    # ------------------------------------------------------------------------------------------------------------------
    def setup(self):
        print("**Widget.setup(self), \tFace Mesh Standardization")

        """    00. Called when the user opens the module the first time and the widget is initialized. """
        ScriptedLoadableModuleWidget.setup(self)

        # 01. Load widget from .ui file.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/FaceStandardization.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # 02. Set scene in MRML widgets.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # 03. Create logic class.
        self.logic = FaceStandardizationLogic()

        # 04. Connections (Single tab)
        self.ui.createSourceLandmarksButton.clicked.connect(self.onCreateSourceLandmarks)
        self.ui.applyStandardizationButton.clicked.connect(self.onApplyStandardization)
        self.ui.sourceMeshSelector.currentNodeChanged.connect(self._onSourceNodeChanged)
        self.ui.sourceLandmarksSelector.currentNodeChanged.connect(self._onLandmarksChanged)
        
        # 04b. Connections (Batch tab)
        self.ui.runBatchButton.clicked.connect(self.onRunBatch)
        self.ui.batchInputDirEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.ui.batchOutputDirEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.ui.batchRefLandmarksEdit.filters = ctk.ctkPathLineEdit().Files
        self.ui.batchRefLandmarksEdit.nameFilters = ["*.fcsv"]
        for w in (
            self.ui.batchInputDirEdit,
            self.ui.batchOutputDirEdit,
            self.ui.batchRefLandmarksEdit,
        ):
            w.connect("currentPathChanged(QString)", self._onBatchPathsChanged)

        # 05. Load bundled reference, show reference image.
        self._loadBundledReference()
        self._showReferenceImage()
        self._updateButtonStates()

    # ------------------------------------------------------------------------------------------------------------------
    def _onBatchPathsChanged(self, *_args):
        self._updateButtonStates()

    # ------------------------------------------------------------------------------------------------------------------
    def cleanup(self):
        """    Called when the application closes and the module widget is destroyed.    """
        print("**Widget.cleanup(self)")
        self.removeObservers()

    # ------------------------------------------------------------------------------------------------------------------
    def enter(self):
        """    Called each time the user opens this module.    """
        print("\n**Widget.enter(self)")
        self._loadBundledReference()
        self._showReferenceImage()

    # ------------------------------------------------------------------------------------------------------------------
    def exit(self):
        """    Called each time the user opens a different module.    """
        print("**Widget.exit(self)")

    # ------------------------------------------------------------------------------------------------------------------
    def _showReferenceImage(self):
        """    Display reference landmarks guide image in the widget panel.    """
        imgPath = _RESOURCES_DIR / "reference_landmarks_guide.png"
        if not imgPath.exists():
            logging.warning("FaceStandardization: reference image not found at %s", imgPath)
            return
        try:
            pixmap = qt.QPixmap(str(imgPath))
            if pixmap.isNull():
                return
            self.ui.referenceImageLabel.setPixmap(pixmap)
            self.ui.referenceImageLabel.setScaledContents(True)
        except Exception as e:
            logging.warning("FaceStandardization: could not load reference image: %s", e)

    # ------------------------------------------------------------------------------------------------------------------
    def _loadBundledReference(self):
        """    Load reference mesh and landmarks from Resources (hidden, for alignment logic only).    """
        scene = slicer.mrmlScene
        paramNode = self.logic.getParameterNode()

        # 01. Try parameterNode (persists in scene, survives reload)
        if paramNode:
            refMesh = paramNode.GetNodeReference("ReferenceMesh")
            refLm = paramNode.GetNodeReference("ReferenceLandmarks")
            if refMesh and refLm and scene.IsNodePresent(refMesh) and scene.IsNodePresent(refLm):
                self._refMesh = refMesh
                self._refLandmarks = refLm
                self._hideReference()
                self._clearReferenceFromSourceSelectors()
                return

        # 02. Try GetFirstNodeByName (fixed names)
        refMesh = scene.GetFirstNodeByName(REF_MESH_NAME)
        refLm = scene.GetFirstNodeByName(REF_LANDMARKS_NAME)
        if refMesh and refLm:
            self._refMesh = refMesh
            self._refLandmarks = refLm
            if paramNode:
                paramNode.SetNodeReferenceID("ReferenceMesh", refMesh.GetID())
                paramNode.SetNodeReferenceID("ReferenceLandmarks", refLm.GetID())
            self._hideReference()
            self._clearReferenceFromSourceSelectors()
            return

        # 03. Load from Resources
        meshPath = _RESOURCES_DIR / "reference_mesh.obj"
        if not meshPath.exists():
            meshPath = _RESOURCES_DIR / "reference_face.ply"
        lmPath = _RESOURCES_DIR / "reference_landmarks.fcsv"

        if not meshPath.exists():
            logging.warning("FaceStandardization: reference mesh not found at %s", meshPath)
            return
        try:
            loaded = slicer.util.loadModel(str(meshPath))
            if isinstance(loaded, (list, tuple)):
                refMesh = loaded[0]
                for extra in loaded[1:]:
                    scene.RemoveNode(extra)
            else:
                refMesh = loaded
            refMesh.SetName(REF_MESH_NAME)
            refMesh.CreateDefaultDisplayNodes()
            if refMesh.GetDisplayNode():
                refMesh.GetDisplayNode().SetVisibility(False)
            self._refMesh = refMesh
        except Exception as e:
            logging.warning("FaceStandardization: failed to load reference mesh: %s", e)
            return

        refLm = None
        if not lmPath.exists():
            logging.warning("FaceStandardization: reference landmarks not found at %s", lmPath)
            return
        try:
            loaded = slicer.util.loadMarkups(str(lmPath))
            refLm = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
            if refLm:
                refLm.SetName(REF_LANDMARKS_NAME)
                if refLm.GetDisplayNode():
                    refLm.GetDisplayNode().SetVisibility(False)
                refLm.SetLocked(True)
                self._refLandmarks = refLm
        except Exception as e:
            logging.warning("FaceStandardization: failed to load reference landmarks: %s", e)
            return

        if self._refMesh and self._refLandmarks and paramNode:
            paramNode.SetNodeReferenceID("ReferenceMesh", self._refMesh.GetID())
            paramNode.SetNodeReferenceID("ReferenceLandmarks", self._refLandmarks.GetID())

        self._hideReference()
        self._clearReferenceFromSourceSelectors()

    # ------------------------------------------------------------------------------------------------------------------
    def _hideReference(self):
        """    Keep reference mesh and landmarks hidden (used only for alignment logic).    """
        if self._refMesh and self._refMesh.GetDisplayNode():
            self._refMesh.GetDisplayNode().SetVisibility(False)
        if self._refLandmarks and self._refLandmarks.GetDisplayNode():
            self._refLandmarks.GetDisplayNode().SetVisibility(False)

    # ------------------------------------------------------------------------------------------------------------------
    def _clearReferenceFromSourceSelectors(self):
        """    Clear source selectors if they point to reference.    """
        if self._refMesh and self.ui.sourceMeshSelector.currentNode() is self._refMesh:
            self.ui.sourceMeshSelector.setCurrentNode(None)
        if self._refLandmarks and self.ui.sourceLandmarksSelector.currentNode() is self._refLandmarks:
            self.ui.sourceLandmarksSelector.setCurrentNode(None)

    # ------------------------------------------------------------------------------------------------------------------
    def _onSourceNodeChanged(self, caller=None, event=None):
        self._updateButtonStates()

    # ------------------------------------------------------------------------------------------------------------------
    def _onLandmarksChanged(self, caller=None, event=None):
        """    Observe fiducial nodes for point add/remove to update button state and set sl_N labels.    """
        srcLm = self.ui.sourceLandmarksSelector.currentNode()
        old, new = self._observedSourceLm, srcLm
        if old and old != new:
            self.removeObserver(old, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._updateButtonStates)
            self.removeObserver(old, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self._onSourceLandmarkPointAdded)
            self.removeObserver(old, slicer.vtkMRMLMarkupsNode.PointRemovedEvent, self._updateButtonStates)
        if new and old != new:
            self.addObserver(new, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._updateButtonStates)
            self.addObserver(new, slicer.vtkMRMLMarkupsNode.PointAddedEvent, self._onSourceLandmarkPointAdded)
            self.addObserver(new, slicer.vtkMRMLMarkupsNode.PointRemovedEvent, self._updateButtonStates)
            self._applySlLabelsToLandmarks(new)
        self._observedSourceLm = srcLm
        self._updateButtonStates()

    # ------------------------------------------------------------------------------------------------------------------
    def _applySlLabelsToLandmarks(self, fiducialNode):
        """    Set labels sl_1, sl_2, ... for all control points in source landmarks.    """
        if not fiducialNode:
            return
        for i in range(fiducialNode.GetNumberOfControlPoints()):
            fiducialNode.SetNthControlPointLabel(i, "sl_%d" % (i + 1))

    # ------------------------------------------------------------------------------------------------------------------
    def _onSourceLandmarkPointAdded(self, caller=None, event=None):
        """    Set label sl_N for newly added source landmark point.    """
        srcLm = self.ui.sourceLandmarksSelector.currentNode()
        if not srcLm:
            return
        n = srcLm.GetNumberOfControlPoints()
        if n > 0:
            srcLm.SetNthControlPointLabel(n - 1, "sl_%d" % n)
        self._updateButtonStates()

    # ------------------------------------------------------------------------------------------------------------------
    def _updateButtonStates(self, caller=None, event=None):
        """    Disable Apply/Run batch when inputs are invalid.    """
        srcMesh = self.ui.sourceMeshSelector.currentNode()
        srcLm = self.ui.sourceLandmarksSelector.currentNode()
        refLm = self._refLandmarks
        valid = (
            all([srcMesh, srcLm, refLm])
            and srcLm.GetNumberOfControlPoints() == 5
        )
        self.ui.applyStandardizationButton.setEnabled(valid)

        inputDir = self._ctkPath(self.ui.batchInputDirEdit)
        refLandmarksPath = self._ctkPath(self.ui.batchRefLandmarksEdit)
        outputDir = self._ctkPath(self.ui.batchOutputDirEdit)
        pathsOk = all([inputDir, refLandmarksPath, outputDir])
        refOk = bool(self._refMesh and self._refLandmarks)
        batchReady = pathsOk and refOk
        self.ui.runBatchButton.setEnabled(batchReady)
        if not batchReady:
            self.ui.batchStatusLabel.text = (
                "Set input and output folders and reference landmarks (.fcsv). "
                "Bundled reference must be loaded (open module once)."
            )
        else:
            self.ui.batchStatusLabel.text = "Click Run batch to standardize all meshes in the input folder."

    # ------------------------------------------------------------------------------------------------------------------
    def onCreateSourceLandmarks(self):
        """    Create new fiducial node for source landmarks, select it for placement.    """
        node = self.logic.createFiducialNode("source_landmarks")
        self.ui.sourceLandmarksSelector.setCurrentNode(node)
        slicer.util.selectModule("Markups")
        slicer.util.infoDisplay(
            "Source landmarks created. Place points on the source mesh in the 3D view.\n"
            "Use Markups toolbar: place mode, or right-click in view."
        )
        self._updateButtonStates()

    # ------------------------------------------------------------------------------------------------------------------
    def onApplyStandardization(self):
        """
        Compute landmark-based similarity transform and apply to mesh(es).
        Validates 5 landmarks; applies to source only or all scene meshes based on Apply to radio.
        """
        print("**Widget.onApplyStandardization(self)")

        srcMesh = self.ui.sourceMeshSelector.currentNode()
        srcLm = self.ui.sourceLandmarksSelector.currentNode()
        refLm = self._refLandmarks

        if not all([srcMesh, srcLm, refLm]):
            slicer.util.warningDisplay(
                "Select source mesh and source landmarks. Reference is loaded automatically."
            )
            return

        if srcLm.GetNumberOfControlPoints() != 5:
            slicer.util.warningDisplay(
                "Exactly 5 landmarks required (got %d). Order: 1=outer right eye, 2=glabella, "
                "3=outer left eye, 4=nose tip, 5=chin." % srcLm.GetNumberOfControlPoints()
            )
            return

        # I. Determine mesh list based on Apply to radio
        allMeshes = self.ui.allMeshesRadio.isChecked()

        if allMeshes:
            meshList = []
            for i in range(slicer.mrmlScene.GetNumberOfNodesByClass("vtkMRMLModelNode")):
                node = slicer.mrmlScene.GetNthNodeByClass(i, "vtkMRMLModelNode")
                if node == srcMesh or node == self._refMesh:
                    continue
                if "Volume Slice" in (node.GetName() or ""):
                    continue
                poly = node.GetPolyData()
                if poly and poly.GetNumberOfPoints() > 0:
                    meshList.append(node)
        else:
            meshList = []

        # II. Run standardization
        runICP = self.ui.icpRefinementCheckBox.isChecked()
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
            with slicer.util.tryWithErrorDisplay("Standardization failed.", waitCursor=True):
                self.logic.standardizeScene(
                    srcMesh,
                    srcLm,
                    meshList,
                    refLm,
                    runICP=runICP,
                    refMeshForICP=self._refMesh if runICP else None,
                    progressCallback=progressCallback,
                )
        finally:
            progressWidget.setVisible(False)
            progressBar.setValue(progressBar.maximum)

        # III. Hide source landmarks, show reference for comparison
        if srcLm.GetDisplayNode():
            srcLm.GetDisplayNode().SetVisibility(False)
        if self._refMesh and self._refMesh.GetDisplayNode():
            self._refMesh.GetDisplayNode().SetVisibility(True)
            self._refMesh.GetDisplayNode().SetColor(1, 0, 0)

        n = 1 + len(meshList)
        msg = "Standardized %d mesh(es)." % n
        if runICP:
            msg += " (with ICP)"
        slicer.util.infoDisplay(msg)
        self._updateButtonStates()

    # ------------------------------------------------------------------------------------------------------------------
    def _ctkPath(self, pathLineEdit):
        """    Path string from ctkPathLineEdit (directories or files).    """
        if pathLineEdit is None:
            return ""
        p = getattr(pathLineEdit, "currentPath", "") or ""
        return (p or "").strip()

    # ------------------------------------------------------------------------------------------------------------------
    def onRunBatch(self):
        """
        Run file-based batch standardization. Load each mesh, apply transform, save, unload.
        """
        print("**Widget.onRunBatch(self)")

        inputDir = self._ctkPath(self.ui.batchInputDirEdit)
        refLandmarksPath = self._ctkPath(self.ui.batchRefLandmarksEdit)
        outputDir = self._ctkPath(self.ui.batchOutputDirEdit)

        if not all([inputDir, refLandmarksPath, outputDir]):
            slicer.util.warningDisplay(
                "Fill all paths: input directory, reference landmarks (.fcsv), output directory."
            )
            return

        if not self._refMesh or not self._refLandmarks:
            slicer.util.warningDisplay("Bundled reference not loaded. Open Single tab first to load reference.")
            return

        runICP = self.ui.batchIcpRefinementCheckBox.isChecked()
        progressWidget = self.ui.batchProgressWidget
        progressBar = self.ui.batchProgressBar

        def progressCallback(current, total, message):
            progressBar.setMaximum(total)
            progressBar.setValue(current)
            progressBar.setFormat("%s (%d/%d)" % (message, current, total))
            slicer.app.processEvents()

        self.ui.runBatchButton.setEnabled(False)
        self.ui.batchStatusLabel.text = "Running batch..."
        progressWidget.setVisible(True)
        progressBar.setValue(0)

        n = None
        try:
            with slicer.util.tryWithErrorDisplay("Batch failed.", waitCursor=True):
                n = self.logic.standardizeFromFiles(
                    Path(inputDir),
                    Path(outputDir),
                    Path(refLandmarksPath),
                    canonicalLandmarks=self._refLandmarks,
                    refMeshForICP=self._refMesh if runICP else None,
                    runICP=runICP,
                    progressCallback=progressCallback,
                )
        finally:
            progressWidget.setVisible(False)
            progressBar.setValue(progressBar.maximum)

        self._updateButtonStates()
        if n is not None:
            msg = "Standardized %d mesh(es) to %s." % (n, outputDir)
            if runICP:
                msg += " (with ICP)"
            slicer.util.infoDisplay(msg)
            self.ui.batchStatusLabel.text = msg

'''=================================================================================================================='''
'''=================================================================================================================='''
#
# FaceStandardizationLogic
#
class FaceStandardizationLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        print("**Logic.__init__(self)")

    # ------------------------------------------------------------------------------------------------------------------
    def createFiducialNode(self, baseName="Landmarks"):
        """    Create a new fiducial node with default display. Returns vtkMRMLMarkupsFiducialNode.    """
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        node.SetName(slicer.mrmlScene.GenerateUniqueName(baseName))
        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
        node.GetDisplayNode().SetSelectedColor(0, 1, 0)
        node.SetLocked(False)
        return node

    # ------------------------------------------------------------------------------------------------------------------
    def _fiducialToVtkPoints(self, fiducialNode, count=None):
        """    Extract control point positions as vtkPoints. If count given, use first count points.    """
        n = fiducialNode.GetNumberOfControlPoints()
        if count is not None:
            n = min(n, count)
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(n)
        pos = [0.0, 0.0, 0.0]
        for i in range(n):
            fiducialNode.GetNthControlPointPosition(i, pos)
            pts.SetPoint(i, pos)
        return pts

    # ------------------------------------------------------------------------------------------------------------------
    def computeTransformToCanonical(self, sourceLandmarksNode, targetLandmarksNode):
        """
        Compute similarity transform: source landmarks -> target (canonical) landmarks.
        Returns vtk.vtkLandmarkTransform.
        Raises ValueError if landmark count != 5.
        """
        nSrc = sourceLandmarksNode.GetNumberOfControlPoints()
        nTgt = targetLandmarksNode.GetNumberOfControlPoints()
        if nSrc != 5 or nTgt != 5:
            raise ValueError(
                "Exactly 5 landmarks required (got %d source, %d target). "
                "Order: 1=outer right eye, 2=glabella, 3=outer left eye, 4=nose tip, 5=chin."
                % (nSrc, nTgt)
            )
        srcPts = self._fiducialToVtkPoints(sourceLandmarksNode, 5)
        tgtPts = self._fiducialToVtkPoints(targetLandmarksNode, 5)
        lmTransform = vtk.vtkLandmarkTransform()
        lmTransform.SetSourceLandmarks(srcPts)
        lmTransform.SetTargetLandmarks(tgtPts)
        lmTransform.SetModeToSimilarity()
        lmTransform.Update()
        return lmTransform

    # ------------------------------------------------------------------------------------------------------------------
    def applyTransformToMesh(self, meshNode, transform):
        """    Apply transform to mesh polydata; bake in place.    """
        poly = meshNode.GetPolyData()
        if not poly or poly.GetNumberOfPoints() == 0:
            raise ValueError("Mesh has no points")
        filt = vtk.vtkTransformPolyDataFilter()
        filt.SetInputData(poly)
        filt.SetTransform(transform)
        filt.Update()
        meshNode.SetAndObservePolyData(filt.GetOutput())
        meshNode.Modified()

    # ------------------------------------------------------------------------------------------------------------------
    def standardizeScene(
        self,
        refMesh,
        refLandmarks,
        meshList,
        canonicalLandmarks,
        runICP=False,
        refMeshForICP=None,
        progressCallback=None,
    ):
        """
        Validate, compute T, apply to reference + all meshes. Bakes transform into geometry.
        meshList: list of vtkMRMLModelNode (excludes reference).
        runICP: if True, run ICP refinement per mesh against refMeshForICP after landmark transform.
        refMeshForICP: target mesh for ICP (canonical reference). Required when runICP=True.
        progressCallback: optional fn(current, total, message) for progress feedback.
        """
        if not refMesh or not refLandmarks or not canonicalLandmarks:
            raise ValueError("Reference mesh, reference landmarks, and canonical landmarks required")
        if runICP and not refMeshForICP:
            raise ValueError("refMeshForICP required when runICP=True")

        allMeshes = [refMesh] + meshList
        n = len(allMeshes)
        total = n * (2 if runICP else 1)
        step = 0

        def report(msg):
            nonlocal step
            step += 1
            if progressCallback:
                progressCallback(step, total, msg)

        transform = self.computeTransformToCanonical(refLandmarks, canonicalLandmarks)
        for mesh in allMeshes:
            name = mesh.GetName() or "mesh"
            report("Processing %s" % name)
            self.applyTransformToMesh(mesh, transform)

        if runICP:
            for mesh in allMeshes:
                name = mesh.GetName() or "mesh"
                report("ICP %s" % name)
                self.icpRefinement(mesh, refMeshForICP)

        logging.info(
            "Standardized %d meshes (ref + %d batch)%s",
            1 + len(meshList),
            len(meshList),
            " + ICP" if runICP else "",
        )

    # ------------------------------------------------------------------------------------------------------------------
    def icpRefinement(self, sourceMeshNode, targetMeshNode, maxIter=150, mode="similarity"):
        """    Refine alignment using VTK ICP. Modifies source mesh in place.    """
        srcPoly = sourceMeshNode.GetPolyData()
        tgtPoly = targetMeshNode.GetPolyData()

        if not srcPoly or srcPoly.GetNumberOfPoints() == 0:
            raise ValueError("Source mesh has no points")
        if not tgtPoly or tgtPoly.GetNumberOfPoints() == 0:
            raise ValueError("Reference mesh has no points")

        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(srcPoly)
        icp.SetTarget(tgtPoly)
        icp.SetMaximumNumberOfIterations(maxIter)
        icp.StartByMatchingCentroidsOn()
        icp.CheckMeanDistanceOn()
        icp.SetMaximumMeanDistance(0.01)

        lm = icp.GetLandmarkTransform()
        if mode == "similarity":
            lm.SetModeToSimilarity()
        else:
            lm.SetModeToRigidBody()

        locator = vtk.vtkCellLocator()
        locator.SetDataSet(tgtPoly)
        locator.BuildLocator()
        icp.SetLocator(locator)

        icp.Update()

        filt = vtk.vtkTransformPolyDataFilter()
        filt.SetInputData(srcPoly)
        filt.SetTransform(icp)
        filt.Update()

        sourceMeshNode.SetAndObservePolyData(filt.GetOutput())
        sourceMeshNode.Modified()

        nIter = icp.GetNumberOfIterations()
        meanDist = icp.GetMeanDistance() if hasattr(icp, "GetMeanDistance") else 0
        logging.info("ICP refinement: %d iterations, mean distance = %g", nIter, meanDist)

    # ------------------------------------------------------------------------------------------------------------------
    def standardizeFromFiles(
        self,
        inputDir,
        outputDir,
        refLandmarksPath,
        canonicalLandmarks,
        runICP=False,
        refMeshForICP=None,
        progressCallback=None,
    ):
        """
        For each mesh in inputDir: load -> compute T (ref landmarks -> canonical) -> apply (+ ICP if enabled) -> save -> unload.
        Returns number of meshes processed.

        Only refLandmarksPath + canonicalLandmarks define the similarity transform; ICP uses refMeshForICP (bundled reference mesh in scene).
        """
        inputDir = Path(inputDir)
        outputDir = Path(outputDir)
        refLandmarksPath = Path(refLandmarksPath)

        if not inputDir.is_dir():
            raise ValueError("Input directory does not exist: %s" % inputDir)
        if not refLandmarksPath.is_file():
            raise ValueError("Reference landmarks file not found: %s" % refLandmarksPath)
        outputDir.mkdir(parents=True, exist_ok=True)

        meshExtensions = {".ply", ".obj", ".vtk", ".stl"}
        meshFiles = [
            f for f in inputDir.iterdir()
            if f.is_file() and f.suffix.lower() in meshExtensions
        ]
        if not meshFiles:
            raise ValueError("No mesh files (.ply, .obj, .vtk, .stl) in input directory")

        refLandmarksNode = self._loadLandmarksFromFile(str(refLandmarksPath))
        if refLandmarksNode.GetNumberOfControlPoints() != 5:
            raise ValueError(
                "Reference landmarks must have exactly 5 points (got %d). "
                "Order: 1=outer right eye, 2=glabella, 3=outer left eye, 4=nose tip, 5=chin."
                % refLandmarksNode.GetNumberOfControlPoints()
            )

        transform = self.computeTransformToCanonical(refLandmarksNode, canonicalLandmarks)
        total = len(meshFiles) * (2 if runICP else 1)
        step = 0

        def report(msg):
            nonlocal step
            step += 1
            if progressCallback:
                progressCallback(step, total, msg)

        processed = 0
        for i, meshFile in enumerate(meshFiles):
            report("Processing %s" % meshFile.name)
            loaded = slicer.util.loadModel(str(meshFile))
            meshNode = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
            if isinstance(loaded, (list, tuple)) and len(loaded) > 1:
                for extra in loaded[1:]:
                    slicer.mrmlScene.RemoveNode(extra)

            poly = meshNode.GetPolyData()
            if not poly or poly.GetNumberOfPoints() == 0:
                logging.warning("Skipping %s: mesh has no points", meshFile.name)
                slicer.mrmlScene.RemoveNode(meshNode)
                continue

            try:
                self.applyTransformToMesh(meshNode, transform)
                if runICP and refMeshForICP:
                    report("ICP %s" % meshFile.name)
                    self.icpRefinement(meshNode, refMeshForICP)

                outPath = outputDir / meshFile.name
                slicer.util.saveNode(meshNode, str(outPath))
                processed += 1
            finally:
                slicer.mrmlScene.RemoveNode(meshNode)

        slicer.mrmlScene.RemoveNode(refLandmarksNode)

        logging.info("Batch: standardized %d meshes to %s", processed, outputDir)
        return processed

    # ------------------------------------------------------------------------------------------------------------------
    def _loadLandmarksFromFile(self, path):
        """    Load fiducials from .fcsv file. Returns vtkMRMLMarkupsFiducialNode.    """
        loaded = slicer.util.loadMarkups(path)
        node = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
        return node

'''=================================================================================================================='''
'''=================================================================================================================='''
#
# FaceStandardizationTest
#
class FaceStandardizationTest(ScriptedLoadableModuleTest):

    def setUp(self):
        slicer.mrmlScene.Clear()

    # ------------------------------------------------------------------------------------------------------------------
    def runTest(self):
        self.setUp()
        self.test_FaceStandardization_LogicExists()

    # ------------------------------------------------------------------------------------------------------------------
    def test_FaceStandardization_LogicExists(self):
        self.delayDisplay("Starting the test")
        logic = FaceStandardizationLogic()
        parameterNode = logic.getParameterNode()
        self.assertIsNotNone(parameterNode)
        self.delayDisplay("Test passed")
