// This is a quick C++ port of the JavaScript SW rasterizer by ryg:
// https://gist.github.com/2486101#file_raster3_ryg6.html

#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>

#include <Windows.h>

using namespace std;

typedef unsigned char PixelComponent;
typedef unsigned int Pixel;

#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))


class CApp {
private:
	static const int blocksize = 8;
	int canvasWidth;
	int canvasHeight;
	int canvasWBlocks;
	int canvasHBlocks;
	vector<PixelComponent> data;
	vector<bool> block_full;

	static double random()
	{
		// Custom random number generator...
		static unsigned int seed1 = 0x23125253;
		static unsigned int seed2 = 0x7423e823;
		seed1 = 36969 * (seed1 & 65535) + (seed1 >> 16);
		seed2 = 18000 * (seed2 & 65535) + (seed2 >> 16);
		return (double)((seed1 << 16) + seed2) / (double)0xffffffffU;
	}


public:
	CApp()
	{
		canvasWidth = 1024;
		canvasHeight = 512;
		data.resize(canvasWidth * canvasHeight * 4);

		canvasWBlocks = (canvasWidth + blocksize-1) / blocksize;
		canvasHBlocks = (canvasHeight + blocksize-1) / blocksize;
		block_full.resize(canvasWBlocks * canvasHBlocks);
	}

	void render()
	{
		int i, l;

		// clear
		l = data.size();
		for ( i = 3; i < l; i += 4 )
		{
			data[ i ] = 0;
		}

		l = block_full.size();
		for ( i = 0; i < l; i++ )
		{
			block_full[i] = 0;
		}

		// Draw 1000 triangles
		for ( i = 0; i < 1000; i++ ) {
			drawTriangle(
						random() * canvasWidth,
						random() * canvasHeight,
						(Pixel)(random() * 0xffffff),
						random() * canvasWidth,
						random() * canvasHeight,
						(Pixel)(random() * 0xffffff),
						random() * canvasWidth,
						random() * canvasHeight,
						(Pixel)(random() * 0xffffff)
						);
		}
	}

	void drawPixel( int x, int y, PixelComponent r, PixelComponent g, PixelComponent b )
	{
		int offset = ( x + y * canvasWidth ) * 4;

		if ( data[ offset + 3 ] )
			return;

		data[ offset ] = r;
		data[ offset + 1 ] = g;
		data[ offset + 2 ] = b;
		data[ offset + 3 ] = 255;
	}

	void drawRectangle( double x1, double y1, double x2, double y2, Pixel color )
	{
		PixelComponent r = color >> 16 & 255;
		PixelComponent g = color >> 8 & 255;
		PixelComponent b = color & 255;

		int xmin = (int)(MIN( x1, x2 ));
		int xmax = (int)(MAX( x1, x2 ));
		int ymin = (int)(MIN( y1, y2 ));
		int ymax = (int)(MAX( y1, y2 ));

		for ( int y = ymin; y < ymax; y ++ )
		{
			for ( int x = xmin; x < xmax; x ++ )
			{
				drawPixel( x, y, r, g, b );
			}
		}
	}

	void drawTriangle( double ax1, double ay1, Pixel color1, double ax2, double ay2, Pixel color2, double ax3, double ay3, Pixel color3 )
	{
		// http://devmaster.net/forums/topic/1145-advanced-rasterization/

		// 28.4 fixed-point coordinates
		int x1 = (int)( 16.0 * ax1 + 0.5 );
		int x2 = (int)( 16.0 * ax2 + 0.5 );
		int x3 = (int)( 16.0 * ax3 + 0.5 );

		int y1 = (int)( 16.0 * ay1 + 0.5 );
		int y2 = (int)( 16.0 * ay2 + 0.5 );
		int y3 = (int)( 16.0 * ay3 + 0.5 );

		// Deltas
		int dx12 = x1 - x2, dy12 = y2 - y1;
		int dx23 = x2 - x3, dy23 = y3 - y2;
		int dx31 = x3 - x1, dy31 = y1 - y3;

		// Bounding rectangle
		int minx = MAX( ( MIN( x1, MIN ( x2, x3 ) ) + 0xf ) >> 4, 0 );
		int maxx = MIN( ( MAX( x1, MAX ( x2, x3 ) ) + 0xf ) >> 4, canvasWidth );
		int miny = MAX( ( MIN( y1, MIN ( y2, y3 ) ) + 0xf ) >> 4, 0 );
		int maxy = MIN( ( MAX( y1, MAX ( y2, y3 ) ) + 0xf ) >> 4, canvasHeight );

		// Block size, standard 8x8 (must be power of two)
		int q = blocksize;

		// Start in corner of 8x8 block
		minx &= ~(q - 1);
		miny &= ~(q - 1);

		// Constant part of half-edge functions
		int c1 = -dy12 * x1 - dx12 * y1;
		int c2 = -dy23 * x2 - dx23 * y2;
		int c3 = -dy31 * x3 - dx31 * y3;

		// Correct for fill convention
    if (dy12 <= 0 && (dy12 != 0 || dx12 <= 0)) c1--;
    if (dy23 <= 0 && (dy23 != 0 || dx23 <= 0)) c2--;
    if (dy31 <= 0 && (dy31 != 0 || dx31 <= 0)) c3--;

		// Note this doesn't kill subpixel precision, but only because we test for >=0 (not >0).
		// It's a bit subtle. :)
    c1 >>= 4;
    c2 >>= 4;
    c3 >>= 4;

		// Set up min/max corners
		int qm1 = q - 1; // for convenience
		int nmin1 = 0, nmax1 = 0;
		int nmin2 = 0, nmax2 = 0;
		int nmin3 = 0, nmax3 = 0;
		if (dx12 >= 0) nmax1 -= qm1*dx12; else nmin1 -= qm1*dx12;
		if (dy12 >= 0) nmax1 -= qm1*dy12; else nmin1 -= qm1*dy12;
		if (dx23 >= 0) nmax2 -= qm1*dx23; else nmin2 -= qm1*dx23;
		if (dy23 >= 0) nmax2 -= qm1*dy23; else nmin2 -= qm1*dy23;
		if (dx31 >= 0) nmax3 -= qm1*dx31; else nmin3 -= qm1*dx31;
		if (dy31 >= 0) nmax3 -= qm1*dy31; else nmin3 -= qm1*dy31;

		// Loop through blocks
		int linestep = (canvasWidth - q) * 4;
		double scale = 255.0 / (c1 + c2 + c3);

    int cyb1 = c1 + dx12 * miny + dy12 * minx;
    int cyb2 = c2 + dx23 * miny + dy23 * minx;
    int cyb3 = c3 + dx31 * miny + dy31 * minx;

		for ( int y0 = miny; y0 < maxy; y0 += q, cyb1 += q*dx12, cyb2 += q*dx23, cyb3 += q*dx31 )
		{
      int cxb1 = cyb1;
      int cxb2 = cyb2;
      int cxb3 = cyb3;

			for ( int x0 = minx; x0 < maxx; x0 += q, cxb1 += q*dy12, cxb2 += q*dy23, cxb3 += q*dy31 )
			{
				// Edge functions at top-left corner
        int cy1 = cxb1;
        int cy2 = cxb2;
        int cy3 = cxb3;

				// Skip block when at least one edge completely out
				if (cy1 < nmax1 || cy2 < nmax2 || cy3 < nmax3)
					continue;

				// Skip writing full block if it's already fully covered
				int blockX = (x0 / q);
				int blockY = (y0 / q);
				int blockInd = blockX + blockY * canvasWBlocks;
				if (block_full[blockInd])
					continue;

				// Offset at top-left corner
				int offset = (x0 + y0 * canvasWidth) * 4;

				// Accept whole block when fully covered
				if (cy1 >= nmin1 && cy2 >= nmin2 && cy3 >= nmin3)
				{
					for ( int iy = 0; iy < q; iy ++ )
					{
						int cx1 = cy1;
						int cx2 = cy2;

						for ( int ix = 0; ix < q; ix ++ )
						{
							if (!data[offset + 3])
							{
								int u = (int)(cx1 * scale); // 0-255!
								int v = (int)(cx2 * scale); // 0-255!
								data[offset] = u;
								data[offset + 1] = v;
								data[offset + 2] = 0;
								data[offset + 3] = 255;
							}

							cx1 += dy12;
							cx2 += dy23;
							offset += 4;
						}

						cy1 += dx12;
						cy2 += dx23;
						offset += linestep;
					}

					block_full[blockInd] = true;
				}
				else
				{
					// Partially covered block
					for ( int iy = 0; iy < q; iy ++ )
					{
						int cx1 = cy1;
						int cx2 = cy2;
						int cx3 = cy3;

						for ( int ix = 0; ix < q; ix ++ )
						{
							if ( (cx1 | cx2 | cx3) >= 0 && !data[offset+3])
							{
								int u = (int)(cx1 * scale); // 0-255!
								int v = (int)(cx2 * scale); // 0-255!
								data[offset] = u;
								data[offset + 1] = v;
								data[offset + 2] = 0;
								data[offset + 3] = 255;
							}

							cx1 += dy12;
							cx2 += dy23;
							cx3 += dy31;
							offset += 4;
						}

						cy1 += dx12;
						cy2 += dx23;
						cy3 += dx31;
						offset += linestep;
					}
				}
			}
		}
	}

	void saveAsTGA(const char* fileName)
	{
		// Construct a TGA file header (18 bytes)
		unsigned char hdr[18];
		memset(hdr, 0, 18);
		hdr[2] = 2;     // True color
		hdr[12] = canvasWidth & 255;
		hdr[13] = canvasWidth >> 8;
		hdr[14] = canvasHeight & 255;
		hdr[15] = canvasHeight >> 8;
		hdr[16] = 32;   // 32 bpp

		// Write file
		ofstream of(fileName, ios_base::out | ios_base::binary);
		of.write((const char*)hdr, 18);
		of.write((const char*)&data[0], data.size());
		of.close();
	}
};

int main()
{
  static const int nFrames = 500;

	CApp app;

  LARGE_INTEGER tstart, tend, freq;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&tstart);

	// Benchmark the renderer
	for ( int i = 0; i < nFrames; i++ )
	{
		app.render();
	}

  QueryPerformanceCounter(&tend);
  printf("%.3fms / frame\n", 1000.0 * (tend.QuadPart - tstart.QuadPart) / (nFrames * freq.QuadPart));

  app.saveAsTGA("test.tga");

	return 0;
}