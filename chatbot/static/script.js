async function sendMessage() {
    const input = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const message = input.value.trim();

    if (!message) return;

  
    // Tampilkan pesan user
    chatBox.innerHTML += `
        <div class="user-message">
            ${message}
        </div>
    `;

    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    // Loading
    chatBox.innerHTML += `
        <div class="bot-message" id="loading">
            Mengetik...
        </div>
    `;
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        // DETEKSI ENDPOINT SECARA DINAMIS
        // Jika URL browser mengandung 'chatbot-indobert', tembak ke model IndoBERT.
        // Jika tidak, biarkan default menembak ke model E5.
        let endpoint = "/chat";
        if (window.location.pathname.includes("chatbot-indobert")) {
            endpoint = "/chat-indobert";
        }

        const response = await fetch(endpoint, { // Menggunakan variabel 'endpoint'
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });
        if (!response.ok) {
            throw new Error("Server Error");
        }

        const data = await response.json();


        // Hapus loading
        const loading = document.getElementById("loading");
        if (loading) {
            loading.remove();
        }

        // Greeting
        if (data.type === "greeting") {
            chatBox.innerHTML += `
                <div class="bot-message">
                    ${data.message}
                </div>
            `;
        }
        // Jawaban Dataset
        else if (data.type === "answer") {
            chatBox.innerHTML += `
                <div class="bot-message">
                    <strong>Jawaban</strong><br>
                    ${data.answer}
                    <br><br>
                    <strong>Konteks</strong><br>
                    ${data.context}
                </div>
            `;
        }
        // Tidak ditemukan / Error bawaan
        else {
            chatBox.innerHTML += `
                <div class="bot-message">
                    ${data.message}
                </div>
            `;
        }

    } catch (error) {
        const loading = document.getElementById("loading");
        if (loading) {
            loading.remove();
        }

        chatBox.innerHTML += `
            <div class="bot-message">
                Gagal terhubung ke server.
            </div>
        `;
        console.error(error);
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}