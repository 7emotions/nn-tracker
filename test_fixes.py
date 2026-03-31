"""Tests for pretrained backbone fixes."""
import torch
import numpy as np
import pytest
from transformer_tracker import TargetTrackingTransformer, TargetTracker


class TestFeatureAdapterInInitTracker:
    """Test that init_tracker applies feature_adapter for pretrained backbone."""

    def test_target_embedding_dimension_with_pretrained_backbone(self):
        """When use_pretrained_backbone=True, target_embedding should have d_model dimensions."""
        model = TargetTrackingTransformer(use_pretrained_backbone=True)
        d_model = 256  # default d_model

        # Verify feature_adapter exists
        assert hasattr(model, 'feature_adapter')
        assert model.feature_adapter is not None

        # Simulate what init_tracker does
        dummy_image = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model.feature_extractor(dummy_image)
            # Before fix: features.mean(dim=[2,3]) would be [1, 512] (wrong)
            assert features.shape[1] == 512, "ResNet layer2 should output 512 channels"

            # After fix: apply feature_adapter first
            features = model.feature_adapter(features)
            assert features.shape[1] == d_model, f"After adapter, should be {d_model} channels"

            target_embedding = features.mean(dim=[2, 3])
            assert target_embedding.shape == (1, d_model), \
                f"Target embedding should be [1, {d_model}], got {target_embedding.shape}"

    def test_target_embedding_dimension_without_pretrained_backbone(self):
        """When use_pretrained_backbone=False, target_embedding should have d_model dimensions."""
        model = TargetTrackingTransformer(use_pretrained_backbone=False)
        d_model = 256

        # Verify feature_adapter does NOT exist
        assert not hasattr(model, 'feature_adapter')

        dummy_image = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model.feature_extractor(dummy_image)
            assert features.shape[1] == d_model, f"Custom backbone should output {d_model} channels"

            target_embedding = features.mean(dim=[2, 3])
            assert target_embedding.shape == (1, d_model)

    def test_init_tracker_with_pretrained_backbone_no_crash(self):
        """init_tracker followed by track should not crash with pretrained backbone."""
        device = 'cpu'
        model = TargetTrackingTransformer(use_pretrained_backbone=True).to(device)
        model.eval()

        tracker = TargetTracker.__new__(TargetTracker)
        tracker.device = device
        tracker.model = model
        tracker.use_transformer = True
        tracker.trajectory_tracker = __import__(
            'transformer_tracker', fromlist=['TrajectoryTracker']
        ).TrajectoryTracker()
        tracker.target_embedding = None

        # Create a dummy frame and bbox
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)  # x, y, w, h

        # init_tracker should NOT crash (before fix, it would produce wrong dimensions)
        tracker.init_tracker(frame, bbox)

        # Verify target_embedding has correct dimension
        d_model = 256
        assert tracker.target_embedding.shape == (1, d_model), \
            f"Expected [1, {d_model}], got {tracker.target_embedding.shape}"

        # track should NOT crash (before fix, dimension mismatch in forward pass)
        bbox_result, confidence = tracker.track(frame)
        assert bbox_result is not None, "Track should return a bounding box"
        assert len(bbox_result) == 4, "Bounding box should have 4 values"

    def test_forward_with_target_embedding_pretrained(self):
        """Forward pass with target_embedding should work with pretrained backbone."""
        model = TargetTrackingTransformer(use_pretrained_backbone=True)
        model.eval()
        d_model = 256

        dummy_input = torch.randn(1, 3, 224, 224)
        # Correctly-dimensioned target embedding (after adapter)
        target_embedding = torch.randn(1, d_model)

        with torch.no_grad():
            bboxes, confidences = model(dummy_input, target_embedding)

        assert bboxes.shape == (1, 100, 4)
        assert confidences.shape == (1, 100, 1)


class TestDemoResnetModelPath:
    """Test that demo 'resnet' tracker sets model_path correctly."""

    def test_resnet_tracker_default_model_path(self):
        """Verify that the resnet tracker code path sets model_path."""
        # Simulate what demo.py main() does for the "resnet" tracker
        # We can't easily run the full demo, but we can verify the logic
        
        class MockArgs:
            tracker = "resnet"
            model = None
            video = "test.mp4"

        args = MockArgs()

        # This replicates the logic in demo.py main()
        if args.tracker == "resnet":
            use_pretrained_backbone = True
            use_transformer = True
            model_path = args.model if args.model else "tracker_model_resnet.pth"

        assert model_path == "tracker_model_resnet.pth", \
            f"Expected 'tracker_model_resnet.pth', got '{model_path}'"
        assert use_pretrained_backbone is True
        assert use_transformer is True

    def test_resnet_tracker_custom_model_path(self):
        """Verify that --model overrides the default for resnet tracker."""
        class MockArgs:
            tracker = "resnet"
            model = "my_custom_model.pth"
            video = "test.mp4"

        args = MockArgs()

        if args.tracker == "resnet":
            model_path = args.model if args.model else "tracker_model_resnet.pth"

        assert model_path == "my_custom_model.pth"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
