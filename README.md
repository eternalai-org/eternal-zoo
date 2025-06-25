# 🚀 CryptoModels: Sovereign Weights
## Deploy AI Models to CryptoAgents

This guide will help you deploy your AI models to the CryptoAgents platform using decentralized infrastructure. Whether you're a developer or AI enthusiast, you'll learn how to run your models with complete sovereignty - maintaining full control over your AI weights through decentralized storage and private local execution with the EAI (EternalAI) command-line tool.

## 📑 Table of Contents
- [Key Features](#-key-features)
- [Before You Start](#-before-you-start)
- [Getting Started](#️-getting-started)
- [CLI Overview](#-cli-overview)
- [Running Models](#-running-models)
- [Using the API](#-using-the-api)
- [Advanced Usage](#advanced-usage)
- [Additional Information](#-additional-information)
- [Migration Guide](#-migration-guide)
- [Need Help?](#-need-help)

## 🌟 Key Features

### 🚀 What Makes CryptoModels Different from Ollama & LMStudio

**🌐 TRUE Decentralized Model Storage**
- Unlike Ollama/LMStudio that rely on centralized repositories (Hugging Face, GitHub), CryptoAgents uses **IPFS/Filecoin** for permanent, censorship-resistant model distribution
- Models are stored across a distributed network - **no single point of failure or control**
- Access your models even if traditional platforms go down or restrict access

**🔒 Ultimate Privacy with Local Execution**  
- **100% local inference** - your data never touches external servers (unlike cloud AI services)
- **Zero telemetry** - no usage tracking, no model access logs, no data collection
- **Air-gapped capability** - run models completely offline once downloaded

### 🛠️ Additional Capabilities

- **🏛️ Sovereign Weights**: Maintain complete ownership and control over your AI models
- **🛡️ Zero Trust Privacy**: Your prompts, responses, and model usage remain completely private
- **🔗 OpenAI Compatibility**: Use familiar API endpoints with your existing tools
- **👁️ Multi-Model Support**: Works with both text and vision models
- **⚡ Parallel Processing**: Efficient model compression and upload
- **🔄 Automatic Retries**: Robust error handling for network issues
- **📊 Metadata Management**: Comprehensive model information tracking

### Why Sovereign Weights Matter

In an era of increasing AI centralization, EAI puts you back in control:

- **Own Your Models**: Models are stored on decentralized infrastructure, not controlled by any single entity
- **Private by Design**: All inference happens locally on your hardware - no external API calls, no data collection
- **Censorship Resistant**: Decentralized storage ensures your models remain accessible regardless of platform policies
- **Vendor Independence**: Break free from proprietary AI services and their limitations

##  Before You Start

### Prerequisites
1. A model in `gguf` format (compatible with `llama.cpp`)
2. A [Lighthouse](https://lighthouse.storage/) account and API key

### System Requirements
- macOS or Linux operating system
- Sufficient RAM for your chosen model (see model specifications below)
- Stable internet connection for model uploads

## 🛠️ Getting Started

### Installation

#### For macOS:
```bash
bash mac.sh
```
> **Note**: You'll need `llama.cpp.rb` in the same directory as `mac.sh`

### Setting Up Your Environment

1. Activate the virtual environment:
```bash
source local_ai/bin/activate
```
> **Remember**: Activate this environment each time you use the `eai` tools

2. Verify your installation:
```bash
eai --version
```

## 📖 CLI Overview

EAI uses a structured command hierarchy for better organization. All model operations are grouped under the `model` subcommand:

```bash
# Model operations
eai model run --hash <hash>           # Run a model server
eai model run <model-name>            # Run a preserved model (e.g., qwen3-1.7b)
eai model stop                        # Stop the running model server  
eai model status                      # Check which model is running
eai model download --hash <hash>      # Download a model from IPFS
eai model preserve --folder-path <path>  # Upload/preserve a model to IPFS

# General commands
eai --version                         # Show version information
```

### Command Examples

```bash
# Run a preserved model (user-friendly)
eai model run qwen3-1.7b --port 8080

# Run any model by hash
eai model run --hash bafkreiacd5mwy4a5wkdmvxsk42nsupes5uf4q3dm52k36mvbhgdrez422y --port 8080

# Check status
eai model status

# Stop the running model
eai model stop

# Download a model locally
eai model download --hash bafkreiacd5mwy4a5wkdmvxsk42nsupes5uf4q3dm52k36mvbhgdrez422y

# Upload your own model
eai model preserve --folder-path ./my-model-folder --task chat --ram 8.5
```

## 🚀 Running Models

### Available Pre-uploaded Models

We've prepared several models for you to test with. Each model is listed with its specifications and command to run.

#### 🔤 Qwen3 Series
[Learn more about Qwen3](https://qwenlm.github.io/blog/qwen3/)

| Model | Size | RAM | Command |
|-------|------|-----|---------|
| qwen3-embedding-0.6b | 649 MB | 1.16 GB | `eai model run qwen3-embedding-0.6b` |
| qwen3-1.7b | - | - | `eai model run qwen3-1.7b` |
| qwen3-4b | 4.28 GB | 9.5 GB | `eai model run qwen3-4b` |
| qwen3-8b | 6.21 GB | 12 GB | `eai model run qwen3-8b` |
| qwen3-14b | 15.7 GB | 19.5 GB | `eai model run qwen3-14b` |
| qwen3-30b-a3b | 31 GB | 37.35 GB | `eai model run qwen3-30b-a3b` |
| qwen3-32b | - | - | `eai model run qwen3-32b` |

#### 👁️ Gemma3 Series (Vision Support)
[Learn more about Gemma3](https://deepmind.google/models/gemma/gemma-3/)

| Model | Size | RAM | Command |
|-------|------|-----|---------|
| gemma3-4b | 3.16 GB | 7.9 GB | `eai model run gemma3-4b` |
| gemma3-12b | 8.07 GB | 21.46 GB | `eai model run gemma3-12b` |
| gemma3-27b | 17.2 GB | 38.0 GB | `eai model run gemma3-27b` |

## 💻 Using the API

The API follows the OpenAI-compatible format, making it easy to integrate with existing applications.

### Text Chat Example
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
        {"role": "user", "content": "Hello! Can you help me write a Python function?"}
    ],
    "temperature": 0.7,
    "max_tokens": 4096
}'
```

### Vision Chat Example (Gemma models only)
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
       {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What do you see in this image? Please describe it in detail."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/your-image.jpg"
                }
            }
        ]
       }
    ],
    "temperature": 0.7,
    "max_tokens": 4096
}'
```

### Embedding Example

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "input": ["Hello, world!"]
}'
```

## Advanced Usage

### Uploading Custom Models

You can use `eai model preserve` to upload your own `gguf` models downloaded from [Huggingface](https://huggingface.co/) for deploying to the CryptoAgents platform.

#### Model Preparation

The platform now supports multiple model types through the `--task` parameter:

##### For Chat Models (Text Generation)

Use `--task chat` for conversational AI and text generation models.

1. **Download the model**:
   - Go to Huggingface and download your desired `.gguf` model
   - Example: Download [`Qwen3-8B-Q8_0.gguf`](https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q8_0.gguf)

2. **Prepare the folder structure**:
   - Create a new folder with a descriptive name (e.g., `qwen3-8b-q8`)
   - Place the downloaded `.gguf` file inside this folder
   - Rename the file to match the folder name, but **remove the `.gguf` extension**

**Example Structure for Chat Models:**
```
qwen3-8b-q8/              # Folder name
└── qwen3-8b-q8          # File name (no .gguf extension)
```

##### For Embedding Models

Use `--task embed` for text embedding and similarity models.

1. **Download the embedding model**:
   - Go to Huggingface and download your desired embedding model in `.gguf` format
   - Example: Text embedding models like [`Qwen3 Embedding 0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) or specialized embedding models

2. **Prepare the folder structure**:
   - Create a new folder with a descriptive name (e.g., `qwen3-embedding-0.6b-q8`)
   - Place the downloaded `.gguf` file inside this folder
   - Rename the file to match the folder name, but **remove the `.gguf` extension**

**Example Structure for Embedding Models:**
```
qwen3-embedding-0.6b-q8/         # Folder name
└── qwen3-embedding-0.6b-q8     # File name (no .gguf extension)
```

##### For Vision Models (Image-Text-to-Text)

Use `--task chat` for vision models as they are conversational models with image understanding capabilities.

1. **Download the model files**:
   - Go to Huggingface and download both required files:
     - The main model file (e.g., [`gemma-3-4b-it-q4_0.gguf`](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/blob/main/gemma-3-4b-it-q4_0.gguf))
     - The projector file (e.g., [`mmproj-model-f16-4B.gguf`](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/blob/main/mmproj-model-f16-4B.gguf))

2. **Prepare the folder structure**:
   - Create a new folder with a descriptive name (e.g., `gemma-3-4b-it-q4`)
   - Place both downloaded files inside this folder
   - Rename the files to match the folder name, but **remove the `.gguf` extension**
   - Add `-projector` suffix to the projector file

**Example Structure for Vision Models:**
```
gemma-3-4b-it-q4/                    # Folder name
├── gemma-3-4b-it-q4                # Main model file (no .gguf extension)
└── gemma-3-4b-it-q4-projector      # Projector file (no .gguf extension)
```

#### Estimating RAM Requirements

Use the [GGUF parser](https://www.npmjs.com/package/@huggingface/gguf) to estimate RAM usage:

```bash
npx @huggingface/gguf qwen3-8b-q8/qwen3-8b-q8 --context 32768
```

#### Upload Commands

**Basic Upload:**
```bash
export LIGHTHOUSE_API_KEY=your_api_key
eai model preserve --folder-path qwen3-8b-q8
```

**Advanced Upload with Metadata:**
```bash
export LIGHTHOUSE_API_KEY=your_api_key
eai model preserve \
  --folder-path qwen3-8b-q8 \
  --task chat \
  --ram 12 \
  --hf-repo Qwen/Qwen3-8B-GGUF \
  --hf-file Qwen3-8B-Q8_0.gguf \
  --zip-chunk-size 512 \
  --threads 16 \
  --max-retries 20
```

**Upload for Embedding Models:**
```bash
export LIGHTHOUSE_API_KEY=your_api_key
eai model preserve \
  --folder-path qwen3-embedding-0.6b-q8 \
  --task embed \
  --ram 1.16 \
  --hf-repo Qwen/Qwen3-Embedding-0.6B-GGUF \
  --hf-file Qwen3-Embedding-0.6B-Q8_0.gguf
```

#### Upload Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--folder-path` | Folder containing the model files | - | ✅ |
| `--task` | Task type: `chat` for text generation models, `embed` for embedding models | `chat` | ❌ |
| `--ram` | RAM usage in GB at 32768 context length | - | ❌ |
| `--hf-repo` | Hugging Face repository (e.g., `Qwen/Qwen3-8B-GGUF`) | - | ❌ |
| `--hf-file` | Original Hugging Face filename | - | ❌ |
| `--zip-chunk-size` | Compression chunk size in MB | 512 | ❌ |
| `--threads` | Number of compression threads | 16 | ❌ |
| `--max-retries` | Maximum upload retry attempts | 20 | ❌ |

#### Upload Process

The upload process involves several steps:

1. **Compression**: The model folder is compressed using `tar` and `pigz` for optimal compression
2. **Chunking**: Large files are split into chunks (default: 512MB) for reliable uploads
3. **Parallel Upload**: Multiple chunks are uploaded simultaneously for faster transfer
4. **Retry Logic**: Failed uploads are automatically retried up to 20 times
5. **Metadata Generation**: A metadata file is created with upload information and model details
6. **IPFS Storage**: All files are stored on IPFS via Lighthouse.storage

#### Troubleshooting

**Common Issues:**
- **Missing API Key**: Ensure `LIGHTHOUSE_API_KEY` is set in your environment
- **Network Issues**: The system will automatically retry failed uploads
- **Insufficient RAM**: Check the model's RAM requirements before uploading
- **Invalid File Format**: Ensure the model is in GGUF format

## 📚 Additional Information

### Model Format
- We support GGUF files compatible with llama.cpp
- Convert other formats using [llama.cpp conversion tools](https://github.com/ggerganov/llama.cpp)

### Performance Tips
- Choose quantization levels (Q4, Q6, Q8) based on your hardware capabilities
- Higher quantization (Q8) offers better quality but requires more resources
- Lower quantization (Q4) is more efficient but may affect model performance
- Monitor system resources during model operation
- Use appropriate context lengths for your use case

### Security
- All models are stored on IPFS with content addressing
- This ensures model integrity and secure distribution
- API keys are stored securely in environment variables
- Models are verified before deployment

### Best Practices
1. **Model Selection**
   - Choose models based on your hardware capabilities
   - Consider quantization levels for optimal performance
   - Test models locally before deployment

2. **Resource Management**
   - Monitor RAM usage during model operation
   - Adjust context length based on available memory
   - Use appropriate batch sizes for your use case

3. **API Usage**
   - Implement proper error handling
   - Use appropriate timeouts for requests
   - Cache responses when possible
   - Monitor API usage and performance

4. **Deployment**
   - Test models thoroughly before production use
   - Keep track of model versions and CIDs
   - Document model configurations and requirements
   - Regular backups of model metadata

## 🆘 Need Help?

- Visit our website: [eternalai.org](https://eternalai.org)
- Join our community: [Discord](https://discord.gg/YphRKtSFqS)
- Check our documentation for detailed guides and tutorials
- Report issues on our GitHub repository
- Contact support for enterprise assistance
