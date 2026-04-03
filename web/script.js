fetch("results.json")
  .then(res => res.json())
  .then(data => {
    document.getElementById("accuracy").innerText = data.model_accuracy;
    document.getElementById("samples").innerText = data.samples_used;
    document.getElementById("near").innerText = data.avg_duration + " minutes";
  })
  .catch(err => {
    console.error("Error:", err);
  });