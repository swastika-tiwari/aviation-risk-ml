fetch("results.json")
  .then(res => res.json())
  .then(data => {

    // Metrics
    document.getElementById("accuracy").innerText = data.model_accuracy.toFixed(3);
    document.getElementById("precision").innerText = data.metrics.precision.toFixed(3);
    document.getElementById("recall").innerText = data.metrics.recall.toFixed(3);
    document.getElementById("f1").innerText = data.metrics.f1_score.toFixed(3);

    // Confusion Matrix
    const cm = data.confusion_matrix;
    document.getElementById("cm00").innerText = cm[0][0];
    document.getElementById("cm01").innerText = cm[0][1];
    document.getElementById("cm10").innerText = cm[1][0];
    document.getElementById("cm11").innerText = cm[1][1];

    // Summary
    document.getElementById("total").innerText = data.summary.total;
    document.getElementById("conflicts").innerText = data.summary.conflicts;
    document.getElementById("safe").innerText = data.summary.safe;

    // Risk pairs
    const list = document.getElementById("risks");
    data.sample_risks.forEach(pair => {
      const li = document.createElement("li");
      li.innerText = pair;
      list.appendChild(li);
    });

    // Insight
    document.getElementById("insightText").innerText =
      "The system demonstrates strong capability in identifying potential near-miss events using spatial and temporal flight data. High recall ensures most conflict scenarios are detected, making the model suitable for safety-critical aviation monitoring.";

    // Charts

    new Chart(document.getElementById("metricsChart"), {
      type: "bar",
      data: {
        labels: ["Precision", "Recall", "F1 Score"],
        datasets: [{
          data: [
            data.metrics.precision,
            data.metrics.recall,
            data.metrics.f1_score
          ]
        }]
      }
    });

    new Chart(document.getElementById("summaryChart"), {
      type: "pie",
      data: {
        labels: ["Conflicts", "Safe"],
        datasets: [{
          data: [
            data.summary.conflicts,
            data.summary.safe
          ]
        }]
      }
    });

  })
  .catch(err => console.error(err));