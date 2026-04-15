<?xml version = "1.0" encoding = "UTF-8"?>
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
