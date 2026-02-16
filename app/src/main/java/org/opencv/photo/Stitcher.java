package org.opencv.photo;

import java.util.List;
import org.opencv.core.Mat;

public class Stitcher {
    public static final int PANORAMA = 0;
    public static final int SCANS = 1;

    protected final long nativeObj;
    protected Stitcher(long addr) { nativeObj = addr; }
    public long getNativeObjAddr() { return nativeObj; }

    // This is the magic "create" method you were missing
    public static Stitcher create(int mode) {
        return new Stitcher(create_0(mode));
    }

    public static Stitcher create() {
        return new Stitcher(create_1());
    }

    public int stitch(List<Mat> images, Mat pano) {
        long images_mat_nativeObj = org.opencv.utils.Converters.vector_Mat_to_Mat(images).getNativeObjAddr();
        return stitch_0(nativeObj, images_mat_nativeObj, pano.nativeObj);
    }

    // Native C++ calls
    private static native long create_0(int mode);
    private static native long create_1();
    private static native int stitch_0(long nativeObj, long images_mat_nativeObj, long pano_nativeObj);
}
