//
//  CoFutureTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 20.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import XCTest
import SwiftCoroutine

class CoFutureTests2: XCTestCase {
    
    func testTransform() {
        let expectation = XCTestExpectation(description: "Test Transform")
        expectation.expectedFulfillmentCount = 3
        let promise = async { () -> Int in
            sleep(1)
            return 1
        }
        let transformed = promise
            .transformOutput { $0 * 2 }
            .onError { _ in XCTFail() }
            .transformOutput { $0 * 3 }
            .onSuccess { XCTAssertEqual($0, 6) }
            .onSuccess { _ in expectation.fulfill() }
            .transformOutput { $0.description }
            .onCompletion { expectation.fulfill() }
        transformed.onSuccess(on: .global) {
            XCTAssertEqual($0, "6")
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 5)
    }
    
}
