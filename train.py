import argparse
import os
from pathlib import Path
import tensorflow as tf

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

def load_datasets(data_dir, batch_size):
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    test_dir = Path(data_dir) / "test"

    for folder in (train_dir, val_dir):
        if not folder.exists():
            raise FileNotFoundError(
                f"Missing {folder}. Expected data_dir/train/<class> and data_dir/val/<class>."
            )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical"
    )
    class_names = train_ds.class_names
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical",
        class_names=class_names, shuffle=False
    )
    test_ds = None
    if test_dir.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir, image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical",
            class_names=class_names, shuffle=False
        )
    return train_ds, val_ds, test_ds, class_names

def main():
    parser = argparse.ArgumentParser(description="Train EcoVision waste classifier with MobileNetV2.")
    parser.add_argument("--data_dir", default="data/waste", help="Folder containing train/val/test splits.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fine_tune", action="store_true", help="Unfreeze the top layers for a short fine-tuning stage.")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    train_ds, val_ds, test_ds, class_names = load_datasets(args.data_dir, args.batch_size)
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint("models/best_ecovision.keras", save_best_only=True, monitor="val_loss"),
    ]
    Path("models").mkdir(exist_ok=True)
    print("Classes:", class_names)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    if args.fine_tune:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

    model = tf.keras.models.load_model("models/best_ecovision.keras")
    if test_ds is not None:
        loss, accuracy = model.evaluate(test_ds, verbose=1)
        print(f"Test accuracy: {accuracy:.4f} | Test loss: {loss:.4f}")

    model.save("models/ecovision.keras")
    with open("models/class_names.txt", "w", encoding="utf-8") as f:
        f.write("\\n".join(class_names))
    print("Saved model to models/ecovision.keras")

if __name__ == "__main__":
    main()
