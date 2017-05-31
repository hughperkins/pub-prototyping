#include <png++/png.hpp>

#include <iostream>
#include <stdexcept>
using namespace std;

#include <ft2build.h>
#include <freetype/freetype.h>
#include <freetype/ftglyph.h>
#include <freetype/ftoutln.h>
#include <freetype/fttrigon.h>

class Freetype {
public:
   Freetype(const char *fontfilepath);
   ~Freetype();
};

FT_BitmapGlyph createBitmap( FT_Face face, int h, char ch ) {
    FT_Set_Char_Size( face, h << 6, h << 6, 96, 96);
    if(FT_Load_Glyph( face, FT_Get_Char_Index( face, ch ), FT_LOAD_DEFAULT )) {
        throw std::runtime_error("FT_Load_Glyph failed");
    }
    FT_Glyph glyph;
    if(FT_Get_Glyph( face->glyph, &glyph )) {
        throw std::runtime_error("FT_Get_Glyph failed");
    }
    FT_Glyph_To_Bitmap( &glyph, ft_render_mode_normal, 0, 1 );
    FT_BitmapGlyph bitmap_glyph = (FT_BitmapGlyph)glyph;
    return bitmap_glyph;
}

int main(int argc, char * argv[]) {
   const char *fname = "../resources/DejaVuSansMono-Bold.ttf";
   int h = 12;
   // char ch = 'q';

    FT_Library library;
    if (FT_Init_FreeType( &library )) {
        throw std::runtime_error("FT_Init_FreeType failed");
    } 
    FT_Face face;
    if (FT_New_Face( library, fname, 0, &face )) {
        throw std::runtime_error("FT_New_Face failed (there is probably a problem with your font file)");
    }
 
    int totsize = 0;
    const char *word = "write";
    if( argc == 2 ) {
       word = argv[1];
    }
    int wordlen = strlen(word);
    FT_Bitmap *bitmaps = new FT_Bitmap[wordlen];
    FT_BitmapGlyph *bitmapglyphs = new FT_BitmapGlyph[wordlen];
    int maxheight = 0;
    int maxwidth;
    int maxpos = 0;
    int maxneg = 0;
    for( int i  = 0; i < wordlen; i++ ) {
       FT_BitmapGlyph bitmapglyph = createBitmap(face, h, word[i] );
       FT_Bitmap bitmap = bitmapglyph->bitmap;
       bitmapglyphs[i] = bitmapglyph;
       bitmaps[i] = bitmapglyph->bitmap;
       int thiswidth = bitmap.width;
       cout << "width " << thiswidth << " rows " << bitmap.rows << " top " << bitmapglyph->top << " left " <<
          bitmapglyph->left << endl;
       int thisheight = bitmap.rows + (bitmap.rows - bitmapglyph->top );
       int thisneg = bitmap.rows - bitmapglyph->top;
       int  thispos = bitmapglyph->top;
       cout << "thisneg " << thisneg << " thispos " << thispos << endl;
       maxneg = max( maxneg, thisneg );
       maxpos = max( maxpos, thispos );
       maxheight = max(maxheight, thisheight);
       totsize += thiswidth;
    }
    cout << "totsize " << totsize << " " << " " << maxneg << " " << maxpos << " " << (maxneg + maxpos ) << endl;
    //cout << "bitmap rows " << 

    maxheight = maxneg + maxpos;
    png::image<png::rgb_pixel> image(totsize, maxheight);
    int xoffset = 0;
    for( int i = 0; i < wordlen; i++ ) {
       FT_Bitmap bitmap = bitmaps[i];
       FT_BitmapGlyph bitmapglyph = bitmapglyphs[i];
       int thisneg = bitmap.rows - bitmapglyph->top;
       int  thispos = bitmapglyph->top;
    for (size_t y = 0; y < bitmap.rows; y++)
    {
        for (size_t x = 0; x < bitmap.width; x++)
        {
               image[min<int>(maxneg+maxpos - 1,y + maxheight - bitmapglyph->top - maxneg)][x + xoffset] = png::rgb_pixel(bitmap.buffer[x + bitmap.width*y],bitmap.buffer[x + bitmap.width*y],bitmap.buffer[x + bitmap.width*y]);
        }
    }
    xoffset += bitmap.width;
    }
    image.write("test.png");

    FT_Done_Face(face);
    FT_Done_FreeType(library);

   return 0;
}

