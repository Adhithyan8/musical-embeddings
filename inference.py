import numpy as np
import librosa
import model
import torch
import deeplake

device = torch.device("cuda:0")
models = {
            "fcn": model.FCN().to(device),
            # "musicnn": model.Musicnn(dataset=DATASET).to(self.device),
            # "crnn": model.CRNN().to(self.device),
            # "sample": model.SampleCNN().to(self.device),
            # "se": model.SampleCNNSE().to(self.device),
            # "attention": model.CNNSA().to(self.device),
            # "hcnn": model.HarmonicCNN().to(self.device),
        }

input_lengths = {
            "fcn": 29 * 16000,
            # "musicnn": 3 * 16000,
            # "crnn": 29 * 16000,
            # "sample": 59049,
            # "se": 59049,
            # "attention": 15 * 16000,
            # "hcnn": 5 * 16000,
        }
def run(path,key="fcn"):

    fcn_model = models[key]
    S = torch.load('/content/sota-music-tagging-models/models/jamendo/fcn/best_model.pth')
    if 'spec.mel_scale.fb' in S.keys():
      fcn_model.spec.mel_scale.fb = S['spec.mel_scale.fb']
    fcn_model.load_state_dict(S)
    fcn_model.eval()

    input_length = input_lengths[key]

    signal, _ = librosa.core.load(path, sr=SAMPLE_RATE)
    length = len(signal)
    hop = length // 2 - input_length // 2
    # print("length, input_length", length, input_length)
    x = torch.zeros(1, input_length)
    x[0] = torch.Tensor(signal[hop : hop + input_length]).unsqueeze(0)
    # x = torch.Variable(x.to(device))
    # print("x.max(), x.min(), x.mean()", x.max(), x.min(), x.mean())


    out, representation = fcn_model(x.to(device))

    TAGS = np.array(['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer', 'genre---atmospheric', 'genre---indie', 'instrument---electricpiano', 'genre---newage', 'instrument---strings', 'instrument---drums', 'instrument---drummachine', 'genre---techno', 'instrument---guitar', 'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop', 'genre---chillout', 'genre---metal', 'mood/theme---happy', 'genre---lounge', 'genre---reggae', 'genre---popfolk', 'genre---orchestral', 'instrument---acousticguitar', 'genre---poprock', 'instrument---piano', 'genre---trance', 'genre---dance', 'instrument---electricguitar', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical', 'mood/theme---energetic', 'genre---electronic', 'genre---world', 'genre---experimental', 'instrument---violin', 'genre---folk', 'mood/theme---emotional', 
    'instrument---voice', 'instrument---keyboard', 'genre---pop', 
    'instrument---bass', 'instrument---computer', 'mood/theme---film', 
    'genre---triphop', 'genre---jazz', 'genre---funk', 'mood/theme---relaxing'])

    # print(np.array(TAGS)[torch.topk(out, k=10)[1].detach().cpu().numpy()[0]])
    return representation

