
# pip install --upgrade "watson-developer-cloud>=2.4.1"

from watson_developer_cloud import VisualRecognitionV3
import json

# Authentication
visual_recognition = VisualRecognitionV3(
    version='2018-03-19',
    iam_apikey='{iam_api_key}')

# Classify an image
def classify(path):
    try:
        with open(path, 'rb') as images_file:
            classes = visual_recognition.classify(
                images_file,
                threshold='0.83',
            classifier_ids='DefaultCustomModel_1970291170').get_result()
        print(json.dumps(classes, indent=2))
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

def main():
    pass

if __name__ == "__main__":
    main()
