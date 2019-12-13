//
//  CombineTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 28.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import XCTest
import Combine
import SwiftCoroutine

@available(OSX 10.15, iOS 13.0, *)
class CombineTests: XCTestCase {

    let session = URLSession.shared
    
    func testCombineAwait() {
        let publisher = session.dataTaskPublisher(for: .testImageURL).map(\.data)
        testImageDownload(publisher: publisher)
    }

    func testCompineSubscriptions() {
        let publisher = session.data(for: .testImageURL).map(\.data)
        testImageDownload(publisher: publisher)
    }
    
    private func testImageDownload<P: Publisher>(publisher: P) where P.Output == Data {
        let expectation = XCTestExpectation(description: "Image download")
        coroutine(on: .main) {
            XCTAssertNotNil(try publisher.await())
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 60)
    }
    
}
