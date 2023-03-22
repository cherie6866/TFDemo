package com.example.weatherclassfication;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private String[] neededPermissions = new String[]{
            Manifest.permission.READ_PHONE_STATE
    };
    private Bitmap bitmap;
    private ImageClassifier imageClassifier;
    private TextView tv_flower_detail;
    private ImageView iv_flower;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            imageClassifier = new ImageClassifier(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            Window window = this.getWindow();
            window.clearFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
            window.getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                    | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
            window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
            window.setStatusBarColor(Color.GRAY);

        }
        setContentView(R.layout.activity_main);

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) {
            List<String> permissionList = new ArrayList<>(Arrays.asList(neededPermissions));
            permissionList.add(Manifest.permission.READ_EXTERNAL_STORAGE);
            neededPermissions = permissionList.toArray(new String[0]);
        }

        initView();
        bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.snow_00023);
        iv_flower.setImageBitmap(bitmap);
    }

    private void initView() {
        tv_flower_detail = findViewById(R.id.tv_flower_detail);
        iv_flower = findViewById(R.id.iv_flower);
    }

    private void showToast(String text) {
        Toast.makeText(this, text, Toast.LENGTH_LONG).show();
    }

    // 更换图片
    public void choose_image(View view) {
//        Intent intent = new Intent(Intent.ACTION_PICK);
//        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
//        startActivityForResult(intent, 0);
        String result = imageClassifier.classifyFrame(bitmap);
        tv_flower_detail.setText("识别结果: " + result);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (data == null || data.getData() == null) {
            showToast("获取图片失败");
            return;
        }

        try {
            bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), data.getData());
        } catch (IOException e) {
            e.printStackTrace();
        }

        iv_flower.setImageBitmap(bitmap);
        String result = imageClassifier.classifyFrame(bitmap);
        tv_flower_detail.setText("识别结果: " + result);
    }
}