fetch("results.json")
  .then(res => res.json())
  .then(data => {
    document.getElementById("accuracy").innerText = data.model_accuracy;
    document.getElementById("samples").innerText = data.dataset_size;
    document.getElementById("duration").innerText = data.near_miss_events_detected;
  });