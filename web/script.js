fetch("results.json")
  .then(res => res.json())
  .then(data => {

    console.log("DATA:", data);

    // ---------------------------
    // SAFE ACCESS FUNCTION
    // ---------------------------
    const safe = (val) => (val !== undefined && val !== null) ? val : 0;

    // ---------------------------
    // METRICS
    // ---------------------------
    document.getElementById("accuracy").innerText =
      safe(data.model_accuracy).toFixed(3);

    document.getElementById("precision").innerText =
      safe(data.metrics?.precision).toFixed(3);

    document.getElementById("recall").innerText =
      safe(data.metrics?.recall).toFixed(3);

    document.getElementById("f1").innerText =
      safe(data.metrics?.f1_score).toFixed(3);

    // ---------------------------
    // SUMMARY
    // ---------------------------
    document.getElementById("total").innerText =
      safe(data.summary?.total);

    document.getElementById("conflicts").innerText =
      safe(data.summary?.conflicts);

    document.getElementById("safe").innerText =
      safe(data.summary?.safe);

    // ---------------------------
    // CONFUSION MATRIX
    // ---------------------------
    if (data.confusion_matrix) {
      const cm = data.confusion_matrix;

      document.getElementById("cm00").innerText = cm[0][0];
      document.getElementById("cm01").innerText = cm[0][1];
      document.getElementById("cm10").innerText = cm[1][0];
      document.getElementById("cm11").innerText = cm[1][1];
    }

    // ---------------------------
    // SAMPLE RISKS
    // ---------------------------
    const list = document.getElementById("risks");
    list.innerHTML = "";

    (data.sample_risks || []).forEach(r => {
      const li = document.createElement("li");
      li.innerText = r;
      list.appendChild(li);
    });

  })
  .catch(err => console.error(err));