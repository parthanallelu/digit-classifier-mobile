# flutter_app

Flutter client for the handwritten digit classifier.

## Offline Prediction

The app now supports fully offline prediction using a bundled TensorFlow Lite model.

Run it normally:

```bash
flutter run
```

## Optional Backend URL

The app uses:

- `http://10.0.2.2:5000` on the Android emulator
- `http://127.0.0.1:5000` on desktop

If you run the app on a real phone, pass your computer's LAN IP when starting Flutter:

```bash
flutter run --dart-define=API_BASE_URL=http://192.168.1.5:5000
```

Replace `192.168.1.5` with your computer's IP on the same Wi-Fi network as the phone.
