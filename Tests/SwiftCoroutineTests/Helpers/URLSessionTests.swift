//
//  URLSessionTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 10.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class URLSessionTests: XCTestCase {
    
    func testImageDownload() {
        let exp = expectation(description: "testImageDownload")
        let session = URLSession.shared
        let dataFuture1 = session.dataTaskFuture(for: .testImageURL)
        let dataFuture2 = session.dataTaskFuture(for: URL(string: "https://errorRequest.com")!)
        DispatchQueue.global().startCoroutine {
            XCTAssertNotNil(try? dataFuture1.await().data)
            XCTAssertNil(try? dataFuture2.await().data)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 30)
    }
    
}
