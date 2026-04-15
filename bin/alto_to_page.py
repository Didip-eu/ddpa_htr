#!/usr/bin/env python3
"""
ALTO → Page conversion tool with embedded XSL stylesheet.
"""

import lxml.etree as ET
import sys
import re
from datetime import datetime
from pathlib import Path

USAGE=f"USAGE: {sys.argv[0]} <alto file>.xml [<xslt file>.xsl]\n\nIf no XSL file is provided, the script's embedded stylesheet is used."

if len(sys.argv) < 2 or re.match(r'--?h', sys.argv[1]):
    print(USAGE)
    sys.exit()

xsl="""<?xml version = "1.0" encoding = "UTF-8"?>
<xsl:stylesheet version = "1.0"
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
	xmlns:alto="http://www.loc.gov/standards/alto/ns-v4#"
>
	<xsl:output method="xml"/>
	<xsl:param name="today"/>
	<xsl:param name="source"/>

	<xsl:template match="/">
		<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd">
		        <MetaData>
                		<Creator>prov=Universität Graz/DDH/nprenet@uni-graz.at</Creator>
				<Created>
					<xsl:value-of select="$today"/>
				</Created>
				<Comments>Converted from ALTO file '<xsl:value-of select="$source"/>'</Comments>
			</MetaData>
			<Page>
				<xsl:attribute name="imageFilename">
					<xsl:value-of select="//alto:Description/alto:sourceImageInformation/alto:fileName"/>
				</xsl:attribute>

				<xsl:attribute name="imageWidth">
					<xsl:value-of select="//alto:Layout/alto:Page/@WIDTH"/>
				</xsl:attribute>
				<xsl:attribute name="imageHeight">
					<xsl:value-of select="//alto:Layout/alto:Page/@HEIGHT"/>
				</xsl:attribute>
				<xsl:for-each select="//alto:TextBlock">
					<TextRegion>
						<xsl:attribute name="id">
							<xsl:value-of select="@ID"/>
						</xsl:attribute>
						<xsl:variable name="regionWidth" select="@WIDTH"/>
						<xsl:variable name="regionHeight" select="@HEIGHT"/>
						<Coords>
						<xsl:attribute name="points">
							<xsl:value-of select="@HPOS"/>,<xsl:value-of select="@VPOS"/>
							<xsl:text> </xsl:text>	
							<xsl:value-of select="@HPOS + $regionWidth"/>,<xsl:value-of select="@VPOS"/>
							<xsl:text> </xsl:text>	
							<xsl:value-of select="@HPOS + $regionWidth"/>,<xsl:value-of select="@VPOS + $regionHeight"/>
							<xsl:text> </xsl:text>	
							<xsl:value-of select="@HPOS"/>,<xsl:value-of select="@VPOS + $regionHeight"/>
						</xsl:attribute>
						</Coords>
						<xsl:for-each select="alto:TextLine">
							<TextLine>
								<xsl:attribute name="id">
									<xsl:value-of select="@ID"/>
								</xsl:attribute>
							</TextLine>
							<Coords>
								<xsl:attribute name="points">
									<xsl:value-of select="alto:Shape/alto:Polygon/@POINTS"/>
								</xsl:attribute>
							</Coords>
							<TextEquiv>
								<Unicode>
								<xsl:value-of select="alto:String/@CONTENT"/>
								</Unicode>
							</TextEquiv>
							<Baseline>
								<xsl:attribute name="points">
									<xsl:value-of select="@BASELINE"/>
								</xsl:attribute>
							</Baseline>
						</xsl:for-each>

					</TextRegion>
				</xsl:for-each>
			</Page>
		</PcGts>
	</xsl:template>
</xsl:stylesheet>
"""


#source_file, xsl_sheet = sys.argv[1:]
source_file = sys.argv[1]
xsl_sheet = sys.argv[2] if len(sys.argv) > 2 else None
    

ns={'alto': "http://www.loc.gov/standards/alto/ns-v4#"}

dom = ET.parse(source_file)

# first, rewrite polygon/line coordinates s.t. it is more palatable for XSLT

def coords_to_pairs( coord_str ):
    coord_str = re.sub(r'\s+',' ',coord_str.strip())
    coord = coord_str.split(' ')
    if len(coord)%2:
        raise(ValueError("Even number of coords expected! Abort."))
    pairs = ' '.join([ f"{coord[i]},{coord[i+1]}" for i in range(0, len(coord), 2) ])
    return pairs


root = dom.getroot()
for plg in root.findall('.//alto:Polygon', ns):
    points_str = plg.get('POINTS')
    plg.set('POINTS', coords_to_pairs( points_str ))

for tl in root.findall('.//alto:TextLine', ns):
    points_str = tl.get('BASELINE')
    tl.set('BASELINE', coords_to_pairs( points_str ))

transform = ET.XSLT( ET.parse( xsl_sheet )) if xsl_sheet and Path(xsl_sheet).exists() else ET.XSLT( ET.XML( xsl.encode() ))
newdom = transform(dom, today=ET.XSLT.strparam(str(datetime.now())), source=ET.XSLT.strparam(Path(source_file).name))

ET.indent( newdom, space='\t', level=0)
print(newdom)

