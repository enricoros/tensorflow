/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.View;
import android.widget.ImageView;

import com.squareup.picasso.Picasso;

import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.enr.Searcher;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import static org.tensorflow.demo.R.id.results;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener, Searcher.Listener {
  private static final Logger LOGGER = new Logger();

  // These are the settings for the original v1 Inception model. If you want to
  // use a model that's been produced from the TensorFlow for Poets codelab,
  // you'll need to set IMAGE_SIZE = 299, IMAGE_MEAN = 128, IMAGE_STD = 128,
  // INPUT_NAME = "Mul:0", and OUTPUT_NAME = "final_result:0".
  // You'll also need to update the MODEL_FILE and LABEL_FILE paths to point to
  // the ones you produced.
  private static final int NUM_CLASSES = 1001;
  private static final int INPUT_SIZE = 224;
  private static final int IMAGE_MEAN = 117;
  private static final float IMAGE_STD = 1;
  private static final String INPUT_NAME = "input:0";
  private static final String OUTPUT_NAME = "output:0";

  private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
  private static final String LABEL_FILE =
      "file:///android_asset/imagenet_comp_graph_label_strings.txt";

  private static final boolean SAVE_PREVIEW_BITMAP = false;

  private static final boolean MAINTAIN_ASPECT = true;

  private TensorFlowImageClassifier classifier;

  private Integer sensorOrientation;

  private int previewWidth = 0;
  private int previewHeight = 0;
  private byte[][] yuvBytes;
  private int[] rgbBytes = null;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;

  private Bitmap cropCopyBitmap;

  private boolean computing = false;


  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private Searcher searcher;
  private List<ImageView> searchViews;
  private ResultsView resultsView;

  private BorderedText borderedText;

  private long lastProcessingTimeMs;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    searcher = new Searcher(this);
    searchViews = new ArrayList<>();
  }

  @Override
  public synchronized void onPause() {
    super.onPause();
    if (searcher != null)
      searcher.saveCacheToDisk(this);
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment;
  }

  @Override
  protected int getDesiredPreviewFrameSize() {
    return INPUT_SIZE;
  }

  private static final float TEXT_SIZE_DIP = 10;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP,
        getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    classifier = new TensorFlowImageClassifier();

    try {
      final int initStatus =
          classifier.initializeTensorFlow(
              getAssets(),
              MODEL_FILE,
              LABEL_FILE,
              NUM_CLASSES,
              INPUT_SIZE,
              IMAGE_MEAN,
              IMAGE_STD,
              INPUT_NAME,
              OUTPUT_NAME);
      if (initStatus != 0) {
        LOGGER.e("TF init status != 0: %d", initStatus);
        throw new RuntimeException();
      }
    } catch (final Exception e) {
      throw new RuntimeException("Error initializing TensorFlow!", e);
    }

    searchViews.clear();
    searchViews.add((ImageView) findViewById(R.id.image_view1));
    searchViews.add((ImageView) findViewById(R.id.image_view2));
    searchViews.add((ImageView) findViewById(R.id.image_view3));
    resultsView = (ResultsView) findViewById(results);
    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    final Display display = getWindowManager().getDefaultDisplay();
    final int screenOrientation = display.getRotation();

    LOGGER.i("Sensor orientation: %d, Screen orientation: %d",
        rotation, screenOrientation);

    sensorOrientation = rotation + screenOrientation;

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbBytes = new int[previewWidth * previewHeight];
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);

    frameToCropTransform = ImageUtils.getTransformationMatrix(
        previewWidth, previewHeight,
        INPUT_SIZE, INPUT_SIZE,
        sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    yuvBytes = new byte[3][];

    addCallback(new DrawCallback() {
      @Override
      public void drawCallback(final Canvas canvas) {
        renderDebug(canvas);
      }
    });
  }

  @Override
  public void onImageAvailable(final ImageReader reader) {
    Image image = null;

    try {
      image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      if (computing) {
        image.close();
        return;
      }
      computing = true;

      Trace.beginSection("imageAvailable");

      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);

      final int yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();
      ImageUtils.convertYUV420ToARGB8888(
          yuvBytes[0],
          yuvBytes[1],
          yuvBytes[2],
          rgbBytes,
          previewWidth,
          previewHeight,
          yRowStride,
          uvRowStride,
          uvPixelStride,
          false);

      image.close();
    } catch (final Exception e) {
      if (image != null) {
        image.close();
      }
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }

    rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            resultsView.setResults(results);

            // find images for these terms
            int resultsCount = results.size();
            for (int i = 0; i < searchViews.size(); i++) {
                final ImageView searchView = searchViews.get(i);
                if (i < resultsCount){
                    Classifier.Recognition recognition = results.get(i);
                    String searchTerm = recognition.getTitle().toLowerCase();
                    searcher.findThumbnailsFor(searchTerm, i, ClassifierActivity.this);
                } else {
                  searchView.post(new Runnable() {
                    @Override
                    public void run() {
                      searchView.setImageResource(0);
                      searchView.setVisibility(View.INVISIBLE);
                    }
                  });
                }
            }

            requestRender();
            computing = false;
          }
        });

    Trace.endSection();
  }

  @Override
  public void onThumbnailsFound(String term, int index, ArrayList<String> thumbnailUrls) {
      if (index < 0 || index >= searchViews.size())
          return;
      ImageView searchView = searchViews.get(index);
      // fixing to 0 because consistency is good
      int rndThumbIdx = 0; //new Random().nextInt(thumbnailUrls.size());
      String thumbnailUrl = thumbnailUrls.get(rndThumbIdx);
      Picasso.with(this).load(thumbnailUrl).into(searchView);
      searchView.setVisibility(View.VISIBLE);
  }

  @Override
  public void onThumbnailsSearchError(String term, int index, String errorString) {
      if (index < 0 || index >= searchViews.size())
          return;
      ImageView searchView = searchViews.get(index);
      searchView.setImageResource(0);
      searchView.setVisibility(View.INVISIBLE);
      LOGGER.e("Thumbnail Search Error: " + errorString);
  }

  @Override
  public void onSetDebug(boolean debug) {
    classifier.enableStatLogging(debug);
  }

  private void renderDebug(final Canvas canvas) {
    if (!isDebug()) {
      return;
    }
    final Bitmap copy = cropCopyBitmap;
    if (copy != null) {
      final Matrix matrix = new Matrix();
      final float scaleFactor = 2;
      matrix.postScale(scaleFactor, scaleFactor);
      matrix.postTranslate(
          canvas.getWidth() - copy.getWidth() * scaleFactor,
          canvas.getHeight() - copy.getHeight() * scaleFactor);
      canvas.drawBitmap(copy, matrix, new Paint());

      final Vector<String> lines = new Vector<String>();
      if (classifier != null) {
        String statString = classifier.getStatString();
        String[] statLines = statString.split("\n");
        for (String line : statLines) {
          lines.add(line);
        }
      }

      lines.add("Frame: " + previewWidth + "x" + previewHeight);
      lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
      lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
      lines.add("Rotation: " + sensorOrientation);
      lines.add("Inference time: " + lastProcessingTimeMs + "ms");

      borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
    }
  }
}
