//
//  CoFutureTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 20.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import XCTest
import SwiftCoroutine

class CoFutureTests: XCTestCase {
    
    func testTransform() {
        let expectation = XCTestExpectation(description: "Test Transform")
        let promise = async { () -> Int in
            sleep(1)
            return 1
        }
        var transformed: CoFuture<String>! = promise
            .map { $0 * 2 }
            .map { $0 * 3 }
            .map { $0.description }
        weak var weakTransformed: CoFuture<String>? = transformed
        transformed.notify(queue: .global()) {
            transformed = nil
            XCTAssertNil(weakTransformed)
            XCTAssertEqual(try? $0.get(), "6")
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 5)
    }
    
}
