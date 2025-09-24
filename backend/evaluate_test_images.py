import os
import io
import json
from collections import defaultdict

# Ensure we use the real model
os.environ['USE_MOCK_MODEL'] = os.environ.get('USE_MOCK_MODEL', '0')
from app import app


def get_true_label_from_filename(fname):
    # filenames like 'A_test.jpg', 'space_test.jpg', 'nothing_test.jpg'
    base = os.path.basename(fname)
    if '_' in base:
        label = base.split('_')[0]
    else:
        label = os.path.splitext(base)[0]
    return label.strip().upper()


def evaluate():
    app.testing = True
    client = app.test_client()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    images_dir = os.path.join(repo_root, 'Test Images')
    if not os.path.isdir(images_dir):
        print('Test Images directory not found at', images_dir)
        return

    files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print('No test images found.')
        return

    total = 0
    correct = 0
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    confusions = defaultdict(lambda: defaultdict(int))

    for fname in sorted(files):
        path = os.path.join(images_dir, fname)
        true_label = get_true_label_from_filename(fname)

        with open(path, 'rb') as f:
            img_bytes = f.read()

        data = {
            'image': (io.BytesIO(img_bytes), fname),
            'session_id': 'eval-session'
        }
        resp = client.post('/predict-image', data=data, content_type='multipart/form-data')
        total += 1
        if resp.status_code != 200:
            print(f'{fname}: ERROR status {resp.status_code}')
            continue
        result = resp.get_json()
        pred_label = result.get('predicted_label')
        if pred_label is None:
            pred_label = ''

        pred_label_norm = str(pred_label).strip().upper()

        per_class_total[true_label] += 1
        confusions[true_label][pred_label_norm] += 1
        if pred_label_norm == true_label:
            correct += 1
            per_class_correct[true_label] += 1

    accuracy = correct / total if total else 0.0

    print('\nEvaluation results')
    print('------------------')
    print(f'Total images: {total}')
    print(f'Correct: {correct}')
    print(f'Accuracy: {accuracy:.3f}')

    print('\nPer-class accuracy:')
    for cls in sorted(per_class_total.keys()):
        tot = per_class_total[cls]
        corr = per_class_correct.get(cls, 0)
        print(f'  {cls}: {corr}/{tot} = {corr/tot:.3f}')

    print('\nConfusion sample (true -> predicted counts):')
    for true in sorted(confusions.keys()):
        row = confusions[true]
        items = ', '.join([f'{pred}:{count}' for pred, count in sorted(row.items(), key=lambda x:-x[1])])
        print(f'  {true} -> {items}')


if __name__ == '__main__':
    evaluate()
