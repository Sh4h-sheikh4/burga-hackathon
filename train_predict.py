import argparse
from pathlib import Path

import cv2
import joblib
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_SIZE = (128, 128)


def image_files(folder):
    return sorted(
        path
        for path in Path(folder).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def descriptor(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.resize(image, IMAGE_SIZE)
    return image.reshape(-1).astype("float32") / 255.0


def load_dataset(root, split):
    features = []
    labels = []
    for label in ("fire", "no_fire"):
        folder = Path(root) / split / label
        files = image_files(folder)
        if not files:
            raise SystemExit(f"No images found in {folder}")
        for path in files:
            features.append(descriptor(path))
            labels.append(label)
    return features, labels


def train(args):
    features, labels = load_dataset(args.data, "train")
    model = LinearSVC(class_weight="balanced", max_iter=10000)
    model.fit(features, labels)
    joblib.dump(model, args.model)
    print(f"Saved model to {args.model}")

    val_dir = Path(args.data) / "val"
    if val_dir.exists():
        try:
            val_features, val_labels = load_dataset(args.data, "val")
        except SystemExit:
            print("Validation images not found; skipping validation.")
        else:
            predictions = model.predict(val_features)
            print(f"Validation accuracy: {accuracy_score(val_labels, predictions):.1%}")


def predict(args):
    model = joblib.load(args.model)
    result = model.predict([descriptor(args.image)])[0]
    print(f"Prediction: {result}")


def main():
    parser = argparse.ArgumentParser(description="Train or use a fire/no-fire image classifier.")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--data", default="data")
    train_parser.add_argument("--model", default="fire_model.joblib")
    train_parser.set_defaults(run=train)

    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("image", help="Path to a JPG or other supported image")
    predict_parser.add_argument("--model", default="fire_model.joblib")
    predict_parser.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
