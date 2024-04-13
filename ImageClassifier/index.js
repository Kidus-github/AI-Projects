const classifier = knnClassifier.create();
const webcamElement = document.getElementById("webcam");

let net;

async function app() {
  console.log("Loading mobilnet...");

  net = await mobilenet.load();

  console.log("Loaded model");

  const webcam = await tf.data.webcam(webcamElement);

  const addExample = async (classId) => {
    const img = await webcam.capture();

    const activation = net.infer(img, true);

    classifier.addExample(activation, classId);

    img.dispose();
  };

  document
    .getElementById("biya")
    .addEventListener("click", () => addExample(0));
  document
    .getElementById("beselam")
    .addEventListener("click", () => addExample(1));
  document
    .getElementById("kidus")
    .addEventListener("click", () => addExample(2));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      const activation = net.infer(img, "conv_preds");

      const result = await classifier.predictClass(activation);

      const classes = ["Biya", "Beselam", "Kidus"];

      document.getElementById("console").innerText = `
                prediction: ${classes[result.label]}\n
                probabilty: ${result.confidences[result.label]}
            `;

      img.dispose();
    }

    await tf.nextFrame();
  }
}

app();
