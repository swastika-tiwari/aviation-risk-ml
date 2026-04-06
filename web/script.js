fetch("results.json")
  .then(response => response.json())
  .then(data => {

    console.log("DATA:", data);

    // =============================
    // METRICS
    // =============================
    document.getElementById("accuracy").innerText = data.model_accuracy.toFixed(3);
    document.getElementById("precision").innerText = data.metrics.precision.toFixed(3);
    document.getElementById("recall").innerText = data.metrics.recall.toFixed(3);
    document.getElementById("f1").innerText = data.metrics.f1_score.toFixed(3);

    // =============================
    // CONFUSION MATRIX
    // =============================
    const cm = data.confusion_matrix;

    document.getElementById("tn").innerText = cm[0][0];
    document.getElementById("fp").innerText = cm[0][1];
    document.getElementById("fn").innerText = cm[1][0];
    document.getElementById("tp").innerText = cm[1][1];

    // =============================
    // RISK DISTRIBUTION
    // =============================
    document.getElementById("critical").innerText = data.risk_distribution.Critical;
    document.getElementById("high").innerText = data.risk_distribution.High;
    document.getElementById("medium").innerText = data.risk_distribution.Medium;
    document.getElementById("low").innerText = data.risk_distribution.Low;

    // =============================
    // TOP RISKS
    // =============================
    const list = document.getElementById("topRisks");
    list.innerHTML = "";

    data.top_risks.forEach(item => {
        const li = document.createElement("li");

        li.innerText = `Pair ${item.pair_id} → Risk: ${item.risk_score} (${item.risk_level})`;

        if (item.risk_level === "Critical") li.style.color = "red";
        else if (item.risk_level === "High") li.style.color = "orange";
        else if (item.risk_level === "Medium") li.style.color = "yellow";

        list.appendChild(li);
    });

    // =============================
    // INSIGHT
    // =============================
    document.getElementById("insight").innerText = data.insight;

  })
  .catch(error => {
    console.error("Error:", error);
  });