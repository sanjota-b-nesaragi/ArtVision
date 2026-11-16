"""
ARTVISION - Database Module
SQLite database management for artwork styles, cultural heritage,
interior design recommendations, skill-based learning, and advertising suggestions.
"""

import os
import sqlite3
import json
from typing import Dict, List, Optional, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "artwork.db")


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create and return a database connection with row factory enabled."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: str = DB_PATH, force: bool = False) -> None:
    """Initialize SQLite database tables and seed initial knowledge base."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Art Styles Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS art_styles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            style_name TEXT UNIQUE NOT NULL,
            time_period TEXT NOT NULL,
            origin_region TEXT NOT NULL,
            historical_background TEXT NOT NULL,
            key_characteristics TEXT NOT NULL,
            artistic_techniques TEXT NOT NULL,
            cultural_significance TEXT NOT NULL,
            prominent_artists TEXT NOT NULL,
            iconic_masterpieces TEXT NOT NULL,
            color_philosophy TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 2. Interior Recommendations Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interior_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            style_name TEXT NOT NULL,
            room_type TEXT NOT NULL,
            mood_ambiance TEXT NOT NULL,
            dominant_color_palettes TEXT NOT NULL,
            framing_advice TEXT NOT NULL,
            lighting_requirements TEXT NOT NULL,
            placement_tips TEXT NOT NULL,
            architectural_styles TEXT NOT NULL,
            FOREIGN KEY (style_name) REFERENCES art_styles(style_name)
        );
    """)

    # 3. Skill-Based Learning Guidance Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skill_guidance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            style_name TEXT NOT NULL,
            difficulty_level TEXT NOT NULL, -- Beginner, Intermediate, Advanced
            learning_objectives TEXT NOT NULL,
            required_materials TEXT NOT NULL,
            core_techniques TEXT NOT NULL,
            step_by_step_exercises TEXT NOT NULL,
            improvement_activities TEXT NOT NULL,
            common_pitfalls TEXT NOT NULL,
            FOREIGN KEY (style_name) REFERENCES art_styles(style_name)
        );
    """)

    # 4. Advertising & Marketing Suggestions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS advertising_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            style_name TEXT NOT NULL,
            brand_archetype TEXT NOT NULL,
            industry_suitability TEXT NOT NULL,
            emotional_impact TEXT NOT NULL,
            color_psychology TEXT NOT NULL,
            campaign_themes TEXT NOT NULL,
            marketing_rationale TEXT NOT NULL,
            target_demographic TEXT NOT NULL,
            FOREIGN KEY (style_name) REFERENCES art_styles(style_name)
        );
    """)

    # 5. Classification Logs / History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classification_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            predicted_style TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            all_probabilities TEXT NOT NULL, -- JSON string
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        );
    """)

    conn.commit()

    # Check if art_styles already seeded
    cursor.execute("SELECT COUNT(*) FROM art_styles;")
    count = cursor.fetchone()[0]

    if count == 0 or force:
        if force:
            cursor.execute("DELETE FROM interior_recommendations;")
            cursor.execute("DELETE FROM skill_guidance;")
            cursor.execute("DELETE FROM advertising_suggestions;")
            cursor.execute("DELETE FROM art_styles;")
        seed_data(cursor)
        conn.commit()

    conn.close()


def seed_data(cursor: sqlite3.Cursor) -> None:
    """Populate database with verified academic artwork style records."""

    styles_data = [
        (
            "Impressionism",
            "Late 19th Century (c. 1860s - 1890s)",
            "Paris, France",
            "Emerging in late 19th-century Paris as an open rebellion against the rigid, conservative French Académie des Beaux-Arts. Led by Claude Monet, Pierre-Auguste Renoir, Edgar Degas, and Camille Pissarro, artists abandoned studio convention to paint 'en plein air' (outdoors). They sought to capture the fleeting sensations of light, atmosphere, and modern life in rapidly modernizing Paris and the surrounding countryside.",
            "Short, visible, broken brushstrokes; open composition with emphasis on natural light and its changing qualities; ordinary everyday subject matter; inclusion of movement as a crucial human perception; candid vantage points influenced by the advent of photography and Japanese woodblock prints.",
            "Alla prima (wet-on-wet paint application); broken color technique (placing complementary colors side-by-side to blend optically in the viewer's eye); avoidance of pure black pigment in shadows, instead employing violet, blue, and deep chromatic hues; loose impasto strokes capturing luminosity.",
            "Marked the definitive birth of modern art by liberating painters from photorealistic academic representation. It shifted art's focus from objective historical narrative to subjective sensory perception, laying foundational principles for Post-Impressionism, Fauvism, and modern abstraction.",
            "Claude Monet, Pierre-Auguste Renoir, Edgar Degas, Camille Pissarro, Berthe Morisot, Mary Cassatt, Alfred Sisley",
            "Monet: 'Impression, Sunrise' (1872), 'Water Lilies' series; Renoir: 'Bal du moulin de la Galette' (1876); Degas: 'The Dance Class' (1874); Pissarro: 'Boulevard Montmartre, Spring' (1897)",
            "Warm daylight yellows, sky cobalts, soft cerulean, lilac, violet shadows, emerald greens, and rosy flesh tones. Prioritizes chromatic harmony over high-contrast monochromatic values."
        ),
        (
            "Cubism",
            "Early 20th Century (c. 1907 - 1920s)",
            "Paris, France / Catalonia, Spain",
            "Pioneered between 1907 and 1914 by Pablo Picasso and Georges Braque in Montmartre, Paris. Deeply influenced by Paul Cézanne's geometric simplification and African tribal sculptures. Cubism shattered the singular Renaissance linear perspective that had dominated Western art for five hundred years, replacing it with a conceptual vision of objects analyzed from multiple simultaneous viewpoints.",
            "Deconstruction of three-dimensional subjects into geometric planes and facets; flattening of traditional pictorial depth; multi-perspective representation (simultaneous front, side, and top views); fragmented forms recombined in abstract spaces; division into Analytical Cubism (monochromatic, complex dissection) and Synthetic Cubism (collage, mixed textures, flatter patterns).",
            "Faceted planar dissection; passage (deliberate blending of overlapping planes); collage and papier collé (integrating newspaper clippings, fabric, and sand); trompe-l'œil text lettering; monochromatic palette reduction to emphasize structural anatomy over decorative color.",
            "Arguably the single most revolutionary visual art movement of the 20th century. It permanently dismantled classical Renaissance illusionism, directly sparking Constructivism, Futurism, De Stijl, Purism, and non-objective abstract art.",
            "Pablo Picasso, Georges Braque, Juan Gris, Fernand Léger, Robert Delaunay, Sonia Delaunay, Albert Gleizes",
            "Picasso: 'Les Demoiselles d'Avignon' (1907), 'Guernica' (1937), 'Girl with a Mandolin' (1910); Braque: 'Houses at l'Estaque' (1908), 'Violin and Candlestick' (1910); Juan Gris: 'Portrait of Pablo Picasso' (1912)",
            "Analytical phase: Earthy ochres, umbers, siennas, slate grays, muted browns, and charcoal. Synthetic phase: Striking primary accents against newsprint grays, sepia, and matte black."
        ),
        (
            "Abstract",
            "Early to Mid 20th Century (c. 1910 - 1960s)",
            "Munich, Germany / Russia / New York, USA",
            "Developed progressively through the early 20th century by pioneers like Wassily Kandinsky, Hilma af Klint, Kazimir Malevich, and Piet Mondrian, later evolving into Abstract Expressionism in postwar New York with Jackson Pollock, Mark Rothko, and Willem de Kooning. It discarded figurative representation to explore the autonomous spiritual, musical, and emotional capacity of pure visual elements.",
            "Total or near-total departure from recognizable physical reality; composition structured entirely through color fields, geometric configurations, dynamic gestural linework, and textured surfaces; rejection of figurative narrative in favor of direct psychological, metaphysical, or aesthetic resonance.",
            "Action painting and dynamic gestural dripping (Pollock); color-field staining onto raw canvas (Rothko, Frankenthaler); geometric reduction and strict neoplastic grid systems (Mondrian); non-objective automatism and synesthetic color-sound translation (Kandinsky); heavy palette knife impasto.",
            "Reframed painting not as an external window onto nature, but as an independent, self-contained sensory object. Abstract art redefined art theory, inspiring modern architecture (Bauhaus), industrial design, typography, and contemporary conceptual art worldwide.",
            "Wassily Kandinsky, Piet Mondrian, Jackson Pollock, Mark Rothko, Kazimir Malevich, Hilma af Klint, Joan Mitchell, Franz Kline",
            "Kandinsky: 'Composition VII' (1913); Mondrian: 'Composition with Red, Blue and Yellow' (1930); Pollock: 'Number 1A' (1948); Malevich: 'Black Square' (1915); Rothko: 'No. 61 (Rust and Blue)' (1953)",
            "Dynamic extremes: either pure primary triads (red, yellow, blue with white and black) in Neoplasticism, meditative luminous color fields (crimson, deep navy, sienna), or raw, high-contrast gestural monochrome."
        ),
        (
            "Surrealism",
            "Interwar Era (c. 1924 - 1950s)",
            "Paris, France (Global influence)",
            "Founded officially in Paris in 1924 by writer André Breton with the 'Surrealist Manifesto'. Spurred by the psychological trauma of World War I and Sigmund Freud's psychoanalytic theories of the subconscious and dream interpretation. Surrealists sought to liberate the human imagination from the tyrannical constraints of rationalism, societal taboos, and bourgeois logic.",
            "Juxtaposition of utterly disparate and contradictory objects; hyper-realistic academic precision applied to dreamscapes and hallucinations; metamorphosis of human and inanimate anatomy; melted landscapes, floating objects, and distorted scale; automatic drawing releasing subconscious impulses.",
            "Psychic automatism (painting or drawing without conscious aesthetic control); exquisite corpse games; frottage (pencil rubbing over textured surfaces); grattage (scraping wet paint); decalcomania; trompe-l'œil veristic representation of irrational, impossible scenarios.",
            "Radically expanded the scope of human psychology explored in visual media, literature, fashion, and cinema (Salvador Dalí and Luis Buñuel's 'Un Chien Andalou'). Continues to exert massive influence on modern advertising, fantasy illustration, digital concept art, and magical realist storytelling.",
            "Salvador Dalí, René Magritte, Max Ernst, Joan Miró, Frida Kahlo, Giorgio de Chirico, Yves Tanguy, Leonora Carrington",
            "Dalí: 'The Persistence of Memory' (1931), 'Swans Reflecting Elephants' (1937); Magritte: 'The Son of Man' (1964), 'The Treachery of Images' (1929); Ernst: 'The Elephant Celebes' (1921); De Chirico: 'The Mystery and Melancholy of a Street' (1914)",
            "Unearthly, high-saturation atmospheric palettes: twilight cobalt, desert terracotta, surreal emerald green, bone white, burnt ochre, and deep abyss black under uncanny golden hour illumination."
        )
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO art_styles (
            style_name, time_period, origin_region, historical_background,
            key_characteristics, artistic_techniques, cultural_significance,
            prominent_artists, iconic_masterpieces, color_philosophy
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, styles_data)

    # Interior Recommendations Data
    interior_data = [
        (
            "Impressionism",
            "Living Rooms, Sunrooms, Master Bedrooms, Dining Areas",
            "Calm, airy, luminous, natural, and romantically serene.",
            "Warm creams, soft sky blues, sage greens, champagne, pastel lavenders, and natural oak wood tones.",
            "Classic ornate gilt frames, warm antiqued gold, or distressed whitewashed wood to complement the gentle luminosity.",
            "Abundant natural daylight; warm diffused ambient lighting (2700K - 3000K). Avoid harsh fluorescent backlights.",
            "Position near floor-to-ceiling windows, reading nooks, or centered above an elegant linen sofa or fireplace mantle.",
            "French Country, Traditional, Transitional, Coastal, Shabby Chic, and Scandinavian Neo-Classic."
        ),
        (
            "Cubism",
            "Executive Offices, Architectural Lofts, Modern Libraries, Dens",
            "Intellectual, analytical, structured, bold, and architecturally avant-garde.",
            "Slate gray, matte black, warm terracotta, raw brass, walnut browns, and muted olive or cream.",
            "Narrow matte black aluminium frames, deep shadowboxes, or raw brushed bronze minimalist profile.",
            "Directional spotlights or recessed ceiling gimbals (3500K - 4000K) casting crisp linear shadows that accent structural facets.",
            "Anchor as a central commanding focal point on an exposed concrete or brick accent wall, or above a structured credenza.",
            "Industrial, Bauhaus, Mid-Century Modern, Brutalist, and Contemporary Urban Loft."
        ),
        (
            "Abstract",
            "Contemporary Living Spaces, Double-Height Foyers, Minimalist Lounges",
            "Dynamic, emotionally evocative, sophisticated, and striking.",
            "Crisp gallery white, charcoal, deep indigo, with vibrant accent splashes (cadmium orange, mustard yellow, cobalt).",
            "Frameless gallery-wrapped stretched canvas, or modern slim floating frames in matte black, bleached maple, or anodized bronze.",
            "Museum-grade wall wash track lights (3000K, high CRI 95+) providing even illumination across large textured surfaces.",
            "Large-scale horizontal or vertical statement canvas mounted at eye level (center at 57-60 inches from floor). Needs negative wall space around it.",
            "Modern Minimalist, Japandi, High-Tech Modern, Contemporary Penthouse, and Art Gallery Aesthetic."
        ),
        (
            "Surrealism",
            "Creative Studios, Private Lounges, Home Theaters, Eclectic Salons",
            "Mysterious, thought-provoking, enigmatic, conversational, and deeply dramatic.",
            "Deep jewel tones (emerald, sapphire, amethyst), midnight navy, burnished copper, velvety black, and ochre accents.",
            "Ornate dark walnut, burnished antiqued bronze, or glossy lacquer floating frames to heighten theatricality.",
            "Dimmable accent accent sconces and mood-setting pinspots highlighting curious details and depth.",
            "Feature in conversational gathering zones, intimate cocktail lounges, or entryway galleries where guests can inspect intricate details.",
            "Eclectic, Dark Academia, Art Deco, Neo-Baroque, and Maximalist Luxury."
        )
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO interior_recommendations (
            style_name, room_type, mood_ambiance, dominant_color_palettes,
            framing_advice, lighting_requirements, placement_tips, architectural_styles
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, interior_data)

    # Skill-Based Learning Guidance Data
    skill_data = [
        # Impressionism
        (
            "Impressionism",
            "Beginner",
            "Understand color temperature, learn wet-on-wet brush handling, and capture natural sunlight on basic landscape forms.",
            "Heavy body acrylics or water-soluble oils; filbert and flat synthetic brushes (#4, #8); stretched canvas board (9x12); palette knife.",
            "Broken color application, dabbed brush marks, eliminating pure black, mixing clean vibrant secondary colors.",
            "Exercise 1: Paint a 5-step color temperature study of a single apple under direct warm sunlight and cool blue shadow.\nExercise 2: 30-minute outdoor sketch of a tree canopy using only rapid dabs of 4 colors (ultramarine, lemon yellow, burnt sienna, titanium white).",
            "Daily 15-minute quick-paint plein air study; keep a visual color diary recording sunrise and sunset sky gradients.",
            "Over-blending strokes until colors turn muddy gray; trying to render minute photographic details rather than atmospheric light impression."
        ),
        (
            "Impressionism",
            "Intermediate",
            "Master optical color mixing, atmospheric perspective, complex water reflections, and dynamic outdoor figuration.",
            "Professional artist-grade oil paints; primed linen panels; hog bristle filberts and brights; odorless mineral spirits; walnut medium.",
            "Scumbling, wet-into-wet impasto, atmospheric color temperature shifts, value compression in high-key daylight lighting.",
            "Exercise 1: Water reflection study capturing moving ripples and sky reflection with rhythmic directional brushstrokes.\nExercise 2: Garden landscape painting completed strictly en plein air in under 90 minutes to capture fleeting cloud shadows.",
            "Study Monet's Rouen Cathedral series to understand how value and color shift with time of day; copy a master study of Camille Pissarro.",
            "Relying on photographs with blown-out highlights; neglecting the cool temperature of ambient ambient sky shadows."
        ),
        (
            "Impressionism",
            "Advanced",
            "Synthesize personal painterly voice with master-level command of chromatic vibration, complex human scenes, and monumental scale.",
            "Custom-stretched Belgian linen canvases; comprehensive oil palette including cadmium and cobalt pigments; assorted palette knives and mops.",
            "Alla prima bravura technique, chromatic vibration through optical side-by-side layering, selective focal clarity against loose periphery.",
            "Exercise 1: Complex multi-figure outdoor gathering (e.g. café terrace or park scene) harmonizing 10+ figures in sunlight and foliage dappling.\nExercise 2: High-contrast sunset marine seascape capturing radiant solar glow and turbulent ocean froth.",
            "Host or join an intensive plein air residency; paint identical monumental scenes across distinct seasonal transitions.",
            "Allowing bravura brushwork to become formulaic or performative without rigorous underlying value structure."
        ),

        # Cubism
        (
            "Cubism",
            "Beginner",
            "Deconstruct simple everyday objects into primary geometric planes, cylinders, and cubes.",
            "Drawing paper (11x14), graphite pencils (2B, 4B, 6B), kneaded eraser, ruler, black fine-liner pens, acrylic gouache or craft acrylics.",
            "Planar analysis, cross-contour drawing, flattening space, observing an object simultaneously from top and side views.",
            "Exercise 1: Draw a coffee mug simultaneously viewed from bird's eye view (circular rim) and horizontal profile (cylinder and handle).\nExercise 2: Paint a three-plane still life of a guitar or fruit bowl using only 3 values of graphite gray and warm sepia.",
            "Carry a sketchbook deconstructing street furniture, vehicles, and buildings into geometric polyhedrons.",
            "Making soft circular curves instead of sharp faceted planar intersections; maintaining standard single-point perspective."
        ),
        (
            "Cubism",
            "Intermediate",
            "Develop Analytical Cubist compositions using grid structures, passage planes, and subtle chromatic modulation.",
            "Acrylics or oils; Bristol board or stretched canvas; masking tape; palette knives; printed typography and newspapers for collage.",
            "Passage (dissolving borders between planes), multi-perspective structural gridding, synthetic papier collé integration, stenciled lettering.",
            "Exercise 1: Create a multi-angle portrait study synthesizing 4 different photographic angles of a face into a coherent faceted mask.\nExercise 2: Construct a mixed-media Synthetic Cubist still life incorporating corrugated cardboard, newsprint fragments, and acrylic paint.",
            "Analyze Braque's and Gris's structural rhythm diagrams; produce analytical line blueprints before applying pigment.",
            "Cluttering the canvas without a strong underlying compositional spine or rhythm; neglecting tonal contrast across adjacent facets."
        ),
        (
            "Cubism",
            "Advanced",
            "Invent complex spatial allegories and monumental multi-layered narratives utilizing synthetic planar deconstruction.",
            "Large-scale primed canvas; mixed media (sand, plaster, oils, collage, metallic leaf); industrial signwriting tools; drafting tools.",
            "Complex spatial ambiguity, textured surface impasto, semiotic visual puns, balance between complete abstraction and recognizable motifs.",
            "Exercise 1: Monumental historical or socio-political statement painting utilizing Cubist fragmentation to convey fractured modern psychology.\nExercise 2: Architectural deconstruction of an urban intersection balancing transparent overlapping geometric planes.",
            "Explore sculptural Cubist reliefs in wood or metal; study Picasso's 1912-1915 cardboard guitars and constructions.",
            "Falling into mere decorative geometric pattern design without conveying conceptual spatial tension."
        ),

        # Abstract
        (
            "Abstract",
            "Beginner",
            "Understand non-representational composition, color harmony, visual weight, rhythm, and mark-making.",
            "Heavyweight watercolor paper (300 gsm), acrylic paints, soft synthetic brushes, palette knives, sponge, masking tape.",
            "Balancing visual weights, wet-in-wet color mixing, hard edge masking vs soft organic gradients, textural mark-making.",
            "Exercise 1: Non-objective emotional color study—create two 8x8 pieces expressing 'Calm' and 'Turbulence' without recognizable shapes.\nExercise 2: Geometric balance study using 3 circles, 2 rectangles, and 1 dynamic diagonal line.",
            "Practice creating 50 distinct marks with various tools (sticks, old credit cards, combs, toothbrushes); build a texture library.",
            "Filling the canvas randomly without visual hierarchy, focal equilibrium, or negative breathing space."
        ),
        (
            "Abstract",
            "Intermediate",
            "Master color field resonance, gestural dynamism, layering translucency, and large-format scale control.",
            "Heavy-duty gallery canvas; acrylic fluid mediums, pouring mediums; professional heavy-body acrylics or oils; large squeegees.",
            "Glazing and veil layering, gestural calligraphic brushwork, high-tension edge relationships, color field depth.",
            "Exercise 1: Three-layer translucent glaze painting creating atmospheric depth using cadmium red, raw sienna, and ultramarine glazes.\nExercise 2: Large-format gestural study executed standing with full-arm momentum, exploring speed and pressure variations.",
            "Study Kandinsky's 'Point and Line to Plane' and Josef Albers' 'Interaction of Color'; execute daily color relativity experiments.",
            "Overworking dynamic gestural strokes until they lose their spontaneity and freshness; poor paint viscosity control."
        ),
        (
            "Abstract",
            "Advanced",
            "Develop a signature conceptual framework, commanding monumental scale, material innovation, and profound emotional resonance.",
            "Unprimed duck canvas or custom panels; raw pigments, rabbit skin glue, polymer resins, oil sticks, industrial squeegees, floor tarps.",
            "Staining raw unprimed canvas, dynamic dripping/splattering with whole-body choreography, monumental reductive color-field meditation.",
            "Exercise 1: Create a 4x5 foot contemplative color-field painting using subtle edge transitions that evoke perceptual luminescence.\nExercise 2: Monumental action painting exploring physical rhythm, controlled spontaneity, and tension between accident and intention.",
            "Curate a cohesive cohesive 6-piece exhibition series around a singular philosophical, musical, or spatial inquiry.",
            "Relying on gimmicks or superficial finishes rather than authentic conceptual depth and structural integrity."
        ),

        # Surrealism
        (
            "Surrealism",
            "Beginner",
            "Harness dream imagery, creative juxtaposition of unrelated objects, and basic trompe-l'œil rendering.",
            "Drawing paper, colored pencils, fine-point pens, soft pastels, magazine clippings for preliminary collage studies.",
            "Uncanny juxtaposition, scale manipulation (making tiny objects gargantuan), metamorphosis of materials (melting or transforming texture).",
            "Exercise 1: Metamorphic object drawing—transform a pocket watch or seashell into an organic sea creature or fruit.\nExercise 2: Surreal collage combining 3 unrelated elements cut from vintage magazines, then rendered realistically in colored pencil.",
            "Keep a bedside dream journal; sketch subconscious imagery immediately upon waking before cognitive editing occurs.",
            "Creating purely whimsical cartoons rather than unsettling, subconscious, or thought-provoking psychological encounters."
        ),
        (
            "Surrealism",
            "Intermediate",
            "Render hyper-realistic textures applied to impossible structures, mastering atmospheric distance and dream lighting.",
            "Smooth gessoed Masonite or fine-weave canvas; oil or alkyd paints; sable round detail brushes (#0, #2); blending mops; glazing mediums.",
            "Veristic trompe-l'œil finish, psychic automatism, frottage and decalcomania backgrounds, unearthly casting of shadows and horizon placement.",
            "Exercise 1: Decalcomania landscape—press wet paint between two boards, pull apart to reveal organic fractal patterns, and paint an uncanny figure inside it.\nExercise 2: Trompe-l'œil floating architectural sphere hovering above a desert plain during eerie blue-hour twilight.",
            "Study Magritte's conceptual paradoxes and Dalí's paranoiac-critical method; master smooth brushless blending transitions.",
            "Allowing brushstrokes to distract from the illusionistic clarity of the impossible scene; lack of coherent lighting logic."
        ),
        (
            "Surrealism",
            "Advanced",
            "Construct complex psychoanalytic narratives and monumental masterworks synthesizing subconscious automatism with flawless classical realism.",
            "Finest gessoed cradled birch panels; traditional multi-layered Venetian oil glaze technique; custom hand-ground pigments and mediums.",
            "Seamless classical glazing, paranoiac-critical double imagery (where one shape simultaneously reads as two distinct subjects), hyper-detailed spatial illusions.",
            "Exercise 1: Large-format double-image masterpiece where a geological canyon landscape simultaneously forms an anatomical portrait.\nExercise 2: Monumental philosophical allegorical dreamscape exploring time, memory, or human identity with metaphysical lighting.",
            "Conduct extensive research into mythology, Jungian archetypes, and subconscious symbolism; write accompanying manifesto essays.",
            "Convoluting the symbolism so densely that visual poetry is lost; sacrificing painterly mastery for shock value."
        )
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO skill_guidance (
            style_name, difficulty_level, learning_objectives, required_materials,
            core_techniques, step_by_step_exercises, improvement_activities, common_pitfalls
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, skill_data)

    # Advertising Suggestions Data
    advertising_data = [
        (
            "Impressionism",
            "The Innocent, The Caregiver, The Lover",
            "Hospitality, Luxury Travel, Organic Wellness, Artisanal Perfumery, High-End Cosmetics, Floral & Winery Brands",
            "Evokes nostalgia, gentle warmth, sensorial delight, romance, authenticity, and peace of mind.",
            "Soft pastel harmonies, warm honey yellows, meadow greens, lavender, and gentle peach tones that reduce visual stress.",
            "'Capture the Moment', 'Natural Radiance', 'Sensory Escapes', 'Timeless Comfort', 'Sunlit Living'",
            "Impressionism creates an instant subconscious association with leisure, natural elegance, and unhurried beauty. Ideal for brands wishing to stand apart from digital noise with organic warmth and humanist craftsmanship.",
            "Affluent leisure consumers, eco-conscious demographics, luxury travelers, wellness enthusiasts (ages 28-65+)."
        ),
        (
            "Cubism",
            "The Creator, The Sage, The Ruler",
            "Architecture & Urban Development, Fintech, High-End Tech Hardware, Analytical Consultancy, Modern Automotive",
            "Conveys cutting-edge intellect, multi-dimensional vision, structural integrity, complexity mastery, and boldness.",
            "Earthy architectural tones (slate gray, burnished copper, charcoal, walnut) energized with sharp metallic or amber accents.",
            "'See From Every Angle', 'Deconstruct Complexity', 'Engineered Vision', 'The Architecture of Tomorrow', 'Multi-Dimensional Clarity'",
            "Signals that a company doesn't just skim the surface—they analyze intricate systems from multiple perspectives. Gives tech and financial services a prestigious, avant-garde gravitas.",
            "Decision-makers, engineers, architects, forward-thinking professionals, fintech investors (ages 25-55)."
        ),
        (
            "Abstract",
            "The Magician, The Outlaw, The Explorer",
            "Creative SaaS & Digital Platforms, Contemporary Fashion, Electronic Music, Modern Fitness & Energy, High-Impact Media",
            "Generates raw emotional excitement, unbounded innovation, individuality, vitality, and forward momentum.",
            "High-contrast dynamic duotones or primary bursts: electric cobalt, vivid neon accents, cadmium red against stark black/white.",
            "'Unleash Pure Potential', 'Beyond Boundaries', 'Feel the Rhythm', 'Redefine Expression', 'Bold by Nature'",
            "Abstract aesthetics bypass analytical cynicism and trigger immediate visceral energy. Free from literal product illustrations, it positions brands as trendsetting cultural catalysts.",
            "Digital natives, creative professionals, Gen Z & Millennials, trendsetters, modern lifestyle adopters (ages 18-40)."
        ),
        (
            "Surrealism",
            "The Visionary, The Jester, The Rebel",
            "Gaming, Film & Streaming, High Fashion, Conceptual Fragrance, Creative Agencies, Craft Distilleries, Immersive Tech (AR/VR)",
            "Triggers intense intrigue, wonder, cognitive fascination, viral shareability, and unforgettable mystery.",
            "Hyper-saturated unearthly contrasts: twilight cobalt, glowing amber, surreal turquoise, velvet obsidian, and metallic accents.",
            "'Question Reality', 'Dream Beyond the Horizon', 'Defy the Ordinary', 'Enter the Unimaginable', 'Curiously Crafted'",
            "Surrealist visuals generate exceptionally high engagement and 'stop-the-scroll' thumb retention in digital feeds. Perfect for challenger brands wanting provocative, memorable campaigns that disrupt conventional expectations.",
            "Creative thinkers, gamers, luxury avant-garde consumers, cultural disruptors, Gen Z & young professionals (ages 18-35)."
        )
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO advertising_suggestions (
            style_name, brand_archetype, industry_suitability, emotional_impact,
            color_psychology, campaign_themes, marketing_rationale, target_demographic
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, advertising_data)


# Query Helper Functions
def get_all_styles(db_path: str = DB_PATH) -> List[str]:
    """Return list of all supported artwork style names."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT style_name FROM art_styles ORDER BY id;")
    rows = cursor.fetchall()
    conn.close()
    return [row["style_name"] for row in rows]


def get_style_details(style_name: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve full cultural and historical record for a specific style."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM art_styles WHERE LOWER(style_name) = LOWER(?);", (style_name,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_interior_recommendations(style_name: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve interior design recommendations for a style."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interior_recommendations WHERE LOWER(style_name) = LOWER(?);", (style_name,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_skill_guidance(style_name: str, difficulty: Optional[str] = None, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve skill-based art learning guides for a style and optional level."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    if difficulty:
        cursor.execute(
            "SELECT * FROM skill_guidance WHERE LOWER(style_name) = LOWER(?) AND LOWER(difficulty_level) = LOWER(?);",
            (style_name, difficulty)
        )
    else:
        cursor.execute(
            "SELECT * FROM skill_guidance WHERE LOWER(style_name) = LOWER(?) ORDER BY id;",
            (style_name,)
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_advertising_suggestions(style_name: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve advertising and marketing suggestions for a style."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM advertising_suggestions WHERE LOWER(style_name) = LOWER(?);", (style_name,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def log_classification(filename: str, predicted_style: str, confidence: float, probabilities: Dict[str, float], notes: str = "", db_path: str = DB_PATH) -> int:
    """Save classification log entry into SQLite history table."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO classification_history (filename, predicted_style, confidence_score, all_probabilities, notes)
        VALUES (?, ?, ?, ?, ?);
    """, (filename, predicted_style, confidence, json.dumps(probabilities), notes))
    entry_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return entry_id


def get_recent_classifications(limit: int = 10, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve recent classification history logs."""
    init_database(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM classification_history ORDER BY id DESC LIMIT ?;", (limit,))
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["all_probabilities"] = json.loads(d["all_probabilities"])
        except Exception:
            pass
        result.append(d)
    return result


if __name__ == "__main__":
    print(f"Initializing database at: {DB_PATH}")
    init_database(force=True)
    styles = get_all_styles()
    print(f"Successfully initialized and seeded database with {len(styles)} styles: {styles}")
