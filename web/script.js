fetch("results.json")
  .then(response => response.json())
  .then(data => {
    console.log("DATA:", data);

    const acc = document.getElementById("accuracy");
    const total = document.getElementById(separation_breach_count["total"]);
    const synthetic = document.getElementById("synthetic");
    const safe = document.getElementById("safe");
    const risks = document.getElementById("risks");

    if (acc) acc.innerText = data.model_accuracy.toFixed(3);

    if (total) total.innerText = data.separation_breach_count.total;
    if (synthetic) synthetic.innerText = data.separation_breach_count.synthetic;
    if (safe) safe.innerText = data.separation_breach_count.safe;

    if (risks) {
      risks.innerText = data.detected_risks.length > 0
        ? data.detected_risks.join(", ")
        : "No risks detected";
    }
  })
  .catch(error => {
    console.error("Error loading JSON:", error);
  });