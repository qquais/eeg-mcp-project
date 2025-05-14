const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const cors = require('cors');
const FormData = require('form-data');
const bodyParser = require('body-parser');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors());
app.use(bodyParser.json());

async function forwardFile(req, res, endpoint, contentType = 'json') {
    try {
        const fileData = req.file;
        if (!fileData) return res.status(400).json({ error: 'No file uploaded' });

        const formData = new FormData();
        formData.append('file', fs.createReadStream(fileData.path));

        const response = await axios.post(`http://localhost:5000${endpoint}`, formData, {
            headers: formData.getHeaders(),
            responseType: contentType === 'stream' ? 'stream' : 'json',
            timeout: 120000
        });

        fs.unlinkSync(fileData.path);

        if (contentType === 'stream') {
            res.setHeader('Content-Type', 'image/png');
            return response.data.pipe(res);
        } else {
            return res.json({ status: 'success', data: response.data });
        }

    } catch (error) {
        console.error(`${endpoint} Error:`, error.message);
        res.status(500).send(`Error processing ${endpoint}`);
    }
}

// Existing MCP Endpoints
app.post('/mcp/eeg', upload.single('file'), (req, res) => forwardFile(req, res, '/read-edf'));
app.post('/mcp/visualize', upload.single('file'), (req, res) => forwardFile(req, res, '/visualize-edf', 'stream'));
app.post('/mcp/features', upload.single('file'), (req, res) => forwardFile(req, res, '/features-edf'));
app.post('/mcp/summary', upload.single('file'), (req, res) => forwardFile(req, res, '/summary-edf'));
app.post('/mcp/export', upload.single('file'), (req, res) => forwardFile(req, res, '/export-edf', 'stream'));
app.post('/mcp/filter', upload.single('file'), (req, res) => forwardFile(req, res, '/filter-edf'));

// ✅ New RAG Query API
app.post('/mcp/query', async (req, res) => {
    const { question } = req.body;
    if (!question) return res.status(400).json({ error: 'Question is required' });

    try {
        const response = await axios.post('http://localhost:11434/api/generate', {
            model: 'mistral',
            prompt: question,
            stream: false
        });

        return res.json({ status: 'success', answer: response.data.response });

    } catch (error) {
        console.error('/mcp/query Error:', error.message);
        res.status(500).send('Error processing query');
    }
});

app.listen(3000, () => {
    console.log('✅ MCP Server running at http://localhost:3000');
});
