from tqdm.notebook import tqdm #use normal tqdm if in shell and not in notebook
import torch
#from crohmeDataset.py import decode_label 
from iamDataset.py import decode_label  #import which ever function for used dataset


def inference_encoder(model, src):
    size = src.size()
    src = model.feature_extractor(src)
    if model.onedim:
        src = src.flatten(2).permute(0, 2, 1)
        src = model.pos_enc_1d(0.1*src)
    else:
        src = model.pos_enc_2d(src)
    return model.encoder(src)

def inference_decoder(model, trg, enc_out, device):
    trg_mask = model.generate_square_subsequent_mask(trg).to(device)
    trg = model.emb(trg)
    trg = model.pos_enc_1d(trg)
    dec_out = model.decoder(trg, enc_out, trg_mask)
    return model.linear(dec_out)


def inference(device, model, data_loader, s2i, i2s, max_len=240):
    model.eval()
    pred = []
    predictions = []

    with torch.no_grad():
        for src, lbl in tqdm(data_loader):
            src, lbl = src.to(device), lbl.to(device)
            enc_out = inference_encoder(model, src.float())
            out_idx = torch.ones(1, 1).fill_(s2i['SOS']).long().to(device)
            for char in range(max_len):
                output = inference_decoder(model, out_idx, enc_out, device)
                out_prediction = torch.argmax(output[:, -1:, :], dim=-1)
                out_idx = torch.cat([out_idx, out_prediction], dim=-1)
                pred.append(out_prediction.item())
                if out_prediction.item() == s2i['EOS']:
                    break
            predictions.append(decode_label(pred, i2s)) 
            pred = []

    return predictions