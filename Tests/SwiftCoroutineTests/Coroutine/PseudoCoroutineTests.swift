//
//  PseudoCoroutineTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 05.04.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class PseudoCoroutineTests: XCTestCase {
    
    func testNestedMain() {
        let result = try? PseudoCoroutine.shared.await(on: DispatchQueue.main) {
            () throws -> Int in
            XCTAssertTrue(Thread.isMainThread)
            return 1
        }
        XCTAssertEqual(result, 1)
    }
    
    func testSuccess() {
        let result = PseudoCoroutine.shared.await(on: DispatchQueue.global()) { () -> Int in
            XCTAssertFalse(Thread.isMainThread)
            return 1
        }
        XCTAssertEqual(result, 1)
    }
    
    func testFailure() {
        do {
            try PseudoCoroutine.shared.await(on: DispatchQueue.global()) {
                XCTAssertFalse(Thread.isMainThread)
                throw CoFutureError.canceled
            }
            XCTFail()
        } catch let error as CoFutureError {
            XCTAssertEqual(error, .canceled)
        } catch {
            XCTFail()
        }
    }
    
}
