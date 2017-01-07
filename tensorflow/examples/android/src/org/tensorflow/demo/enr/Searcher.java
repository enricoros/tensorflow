package org.tensorflow.demo.enr;

import android.content.Context;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Environment;
import android.preference.PreferenceManager;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class Searcher {
    private static final String SP_KEY = "searcher_cache";
    public static final String CACHE_FILE_NAME = "tf-cached-search-data.txt";
    public static final String BING_API_KEY = "ADD YOUR API HERE";
    private final Gson mGson;
    private final HashMap<String, ArrayList<String>> mCache;
    private final ArrayList<String> mTermsInProcess;

    public interface Listener {
        void onThumbnailsFound(String term, int index, ArrayList<String> thumnailUrls);

        void onThumbnailsSearchError(String term, int index, String errorString);
    }

    public Searcher(Context context) {
        mGson = new GsonBuilder().create();
        mCache = loadCacheFromDisk(context);
        mTermsInProcess = new ArrayList<>();
    }

    private HashMap<String, ArrayList<String>> loadCacheFromDisk(Context context) {
        String jsonCache = null;
        // load from shared prefs
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        if (prefs.contains(SP_KEY))
            jsonCache = prefs.getString(SP_KEY, null);
        else {
            try {
                InputStream stream = context.getAssets().open(CACHE_FILE_NAME);
                jsonCache = new BufferedReader(new InputStreamReader(stream)).readLine();
            } catch (IOException e) {
                e.printStackTrace();
                jsonCache = null;
            }
        }
        if (jsonCache == null || jsonCache.isEmpty())
            return new HashMap<>();
        return mGson.fromJson(jsonCache, HashMap.class);
    }

    public void saveCacheToDisk(Context context) {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(context);
        String cacheJson = mGson.toJson(mCache);
        prefs.edit().putString(SP_KEY, cacheJson).apply();

        // write to SD_CARD
        File sdCard = Environment.getExternalStorageDirectory();
        File file = new File(sdCard, CACHE_FILE_NAME);
        try {
            FileOutputStream f = new FileOutputStream(file);
            f.write(cacheJson.getBytes(Charset.forName("UTF-8")));
            f.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void findThumbnailsFor(String term, int index, Listener listener) {
        if (!mTermsInProcess.contains(term)) {
            mTermsInProcess.add(term);
            new AsyncBingSearch(term, index, listener).execute();
        }
    }

    private class AsyncBingSearch extends AsyncTask<Void, Void, Void> {

        private final String mTerm;
        private final int mIndex;
        private final Listener mListener;
        private String mErrorString;
        private ArrayList<String> mThumbnailUrls;

        AsyncBingSearch(String term, int index, Listener listener) {
            mTerm = term;
            mIndex = index;
            mListener = listener;
        }

        @Override
        protected Void doInBackground(Void... params) {
            if (mCache.containsKey(mTerm))
                return null;

            Uri uri = Uri.parse("https://api.cognitive.microsoft.com/bing/v5.0/images/search")
                    .buildUpon()
                    .appendQueryParameter("q", mTerm)
                    .appendQueryParameter("count", "3")
                    .appendQueryParameter("offset", "0")
                    .appendQueryParameter("mkt", "en-us")
                    .appendQueryParameter("safeSearch", "off")
                    .build();

            try {
                URLConnection urlConnection = new URL(uri.toString()).openConnection();
                urlConnection.setRequestProperty("Ocp-Apim-Subscription-Key", BING_API_KEY);

                InputStream in = new BufferedInputStream(urlConnection.getInputStream());
                Scanner s = new Scanner(in).useDelimiter("\\A");
                if (!s.hasNext())
                    return null;

                // parse stream to json
                JSONObject jsonObject = new JSONObject(s.next());

                // scan json for 3 URLs
                mThumbnailUrls = new ArrayList<>();
                JSONArray images = jsonObject.getJSONArray("value");
                for (int i = 0; i < images.length() && i < 3; ++i) {
                    JSONObject jsonBingImage = images.optJSONObject(i);
                    if (jsonBingImage != null) {
                        String thumbnailUrl = jsonBingImage.getString("thumbnailUrl");
                        mThumbnailUrls.add(thumbnailUrl);
                    }
                }
                if (mThumbnailUrls.size() < 3)
                    mThumbnailUrls = null;
            } catch (Exception e) {
                mErrorString = e.getMessage();
            }
            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            // done with the processing
            mTermsInProcess.remove(mTerm);
            // pick from cache, if the search was not conducted
            if (mCache.containsKey(mTerm) && mThumbnailUrls == null) {
                mListener.onThumbnailsFound(mTerm, mIndex, mCache.get(mTerm));
                return;
            }
            // dispatch error, if set
            if (mErrorString != null || mThumbnailUrls == null) {
                mListener.onThumbnailsSearchError(mTerm, mIndex, mErrorString != null ? mErrorString : "no thumbnails");
                return;
            }
            // save to cache and continue
            mCache.put(mTerm, mThumbnailUrls);
            mListener.onThumbnailsFound(mTerm, mIndex, mThumbnailUrls);
        }
    }

}
