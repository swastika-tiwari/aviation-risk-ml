fetch("results.json")
  .then(response => response.json())
  .then(data => {
    console.log("DATA:", data); // 👈 important for debugging

    document.getElementById("accuracy").innerText = data.model_accuracy;
    document.getElementById("samples").innerText = data.samples_used;
    document.getElementById("duration").innerText = data.avg_duration + " minutes";
  })
  .catch(error => {
    console.error("Error loading JSON:", error);
  });