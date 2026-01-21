document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('audioFile');
    const uploadForm = document.getElementById('uploadForm');
    const resultDiv = document.getElementById('result');
    const audioPlayer = document.getElementById('audioPlayer');
    const loadingSpinner = document.querySelector('.spinner');
    
    // Handle file selection
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            // Display audio player for preview
            const url = URL.createObjectURL(file);
            audioPlayer.src = url;
            audioPlayer.style.display = 'block';
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select an audio file first.');
            return;
        }
        
        // Show loading spinner
        loadingSpinner.style.display = 'block';
        resultDiv.innerHTML = '';
        
        const formData = new FormData();
        formData.append('audio_file', file);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            if (response.ok) {
                if (data.prediction) {
                    // Display prediction result
                    resultDiv.innerHTML = `
                        <div class="result-card">
                            <h3>Prediction Result</h3>
                            <p><strong>Disease:</strong> ${data.prediction}</p>
                            <p><strong>Confidence:</strong> ${data.confidence_percent}%</p>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = '<p class="error">Could not make a prediction. Please try another file.</p>';
                }
            } else {
                // Handle error cases
                if (data.error.includes('Authentication required')) {
                    // Redirect to login if not authenticated
                    window.location.href = '/';
                } else {
                    resultDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
                }
            }
        } catch (error) {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            resultDiv.innerHTML = `<p class="error">An error occurred: ${error.message}</p>`;
        }
    });
});